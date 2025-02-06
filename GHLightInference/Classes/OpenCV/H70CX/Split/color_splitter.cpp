#include "color_splitter.hpp"
#include <algorithm>
#include <random>

ColorSplitter::ColorSplitter(const SplitterConfig &config)
        : config_(config),
          strategy_(make_shared<ColorBasedSplitStrategy>()) {
    config_.validate();
    LOGD(LOG_TAG, "创建ColorSplitter实例 - 颜色通道数: %d", color_channels_);
    initMaxSplitCount();

    LOGD(LOG_TAG, "color_mapping_cache_: %d", color_mapping_cache_.size());
}

void ColorSplitter::initColorMappingCache() {
    int max_split_level = getMaxSplitCount();
    LOGD(LOG_TAG, "\n==== 开始初始化颜色映射缓存 ====");
    LOGD(LOG_TAG, "最大分割级别: %d", max_split_level);

    // 递归计算每个分割级别的映射
    for (int level = 0; level <= max_split_level; level++) {
        vector<vector<int>> level_mapping;

        // 创建初始范围
        calculateRangeMappings(0, config_.max_id_range, level, 0, level_mapping);

        color_mapping_cache_[level] = level_mapping;

//        LOGD(LOG_TAG, "第 %d 级分割完成 - 总映射数: %zu", level, level_mapping.size() / 2);
    }

    LOGD(LOG_TAG, "==== 颜色映射缓存��始化完成 ====\n");
}

void ColorSplitter::calculateRangeMappings(
        int start_range,
        int end_range,
        int target_level,
        int current_level,
        vector<vector<int>> &level_mapping
) {
    int range_size = end_range - start_range + 1;

    if (current_level == target_level) {
        // 到达目标级别，记录当前范围内的所有ID和颜色
        for (int id = start_range; id <= end_range; id++) {
            vector<int> id_color_pairs;
            int color = current_level == 0 ? id / (range_size / color_channels_) :
                        (id - start_range) / (range_size / color_channels_);
            color = min(color, color_channels_ - 1);

            id_color_pairs.push_back(id);
            if (color == 0) {
                id_color_pairs.push_back(RED);
            } else {
                id_color_pairs.push_back(GREEN);
            }
            level_mapping.push_back(id_color_pairs);
        }

//        LOGV(LOG_TAG, "级别 %d - 范围[%d, %d] 处理完成",
//             current_level, start_range, end_range);
        return;
    }

    // 继续递归分割
    int sub_range_size = range_size / color_channels_;
//    LOGV(LOG_TAG, "级别 %d - 范围[%d, %d] 分割为 %d 份，份大小 %d",
//         current_level, start_range, end_range, color_channels_, sub_range_size);

    for (int i = 0; i < color_channels_; i++) {
        int sub_start = start_range + i * sub_range_size;
        int sub_end = (i == color_channels_ - 1) ?
                      end_range :
                      sub_start + sub_range_size - 1;

//        LOGV(LOG_TAG, "级别 %d - 子范围 %d: [%d, %d]",
//             current_level, i, sub_start, sub_end);

        calculateRangeMappings(
                sub_start,
                sub_end,
                target_level,
                current_level + 1,
                level_mapping
        );
    }
}

bool ColorSplitter::initialize(const Mat &image, vector<Point2f> points, vector<Mat> &outMats) {
    // 如果是第一次处理图片，执行初始分割
    point_sets_.clear();
    confirmed_points_.clear();
    // 创建初始点集
    PointSet initial_set = PointSet(0, config_.max_id_range);

    for (const auto &item: points) {
        LedPoint p(-1, Point2f(item.x, item.y), 6);
        initial_set.points.push_back(p);
    }

    // 对未确认的集合进行颜色分割
    auto split_results = strategy_->split(initial_set, image);
    strategy_->split_count = 0;
    // 处理分割后的子集合
    for (auto &subset: split_results) {
        LOGV(LOG_TAG, "首次分割 subset = %d - %d", subset.start_range, subset.end_range);
        point_sets_.push_back(subset);
    }
    // 使用Visualizer绘制当前状态
    Visualizer::visualizeSplitState(
            image,
            point_sets_,
            confirmed_points_,
            strategy_->split_count,
            outMats
    );

    LOGD(LOG_TAG, "首次分割 point_sets_ = %d", point_sets_.size());
    return true;  // 首次分割后还需要继续处理
}

bool ColorSplitter::processImage(const Mat &image, vector<Mat> &outMats) {
    try {
        // 验证输入图像
        if (image.empty()) {
            LOGE(LOG_TAG, "Empty input image");
            return false;
        }

        // 如果是第一次处理图片，执行初始分割
        if (point_sets_.empty()) {
            LOGE(LOG_TAG, "未完成初次分割！");
            return false;
        }

        // 对现有的每个未确认点集进行进一步分割
        vector<PointSet> new_sets;
        for (const auto &set: point_sets_) {
            // 如果集合中的点都已确认，跳过该集合
            bool all_confirmed = all_of(set.points.begin(), set.points.end(),
                                        [](const LedPoint &p) { return p.confirmed; });
            if (all_confirmed) continue;

            // 对未确认的集合进行颜色分割
            auto split_results = strategy_->split(set, image);

            // 处理分割后的子集合
            for (auto &subset: split_results) {
                if (!subset.points.empty()) {
                    // 更新点的颜色属性
                    for (auto &point: subset.points) {
                        point.color = point.calculateAverageColor(image);
                    }
                    new_sets.push_back(subset);
                }
            }
        }
        strategy_->split_count++;

        // 使用Visualizer绘制当前状态
        Visualizer::visualizeSplitState(
                image,
                new_sets,
                confirmed_points_,
                strategy_->split_count,
                outMats
        );

        // 更新点集
        point_sets_ = new_sets;

        for (auto &item: point_sets_) {
            if (item.range() <= 2 && item.range() >= item.size()) {
                processSmallSet(item);
            }
        }

        // 计算距离标准差
        float distance_total = 0;
        int count = 0;
        for (size_t i = 0; i < confirmed_points_.size(); i++) {
            for (size_t j = i + 1; j < confirmed_points_.size(); j++) {
                if (abs(confirmed_points_[i].id - confirmed_points_[j].id) == 1) {
                    distance_total += norm(
                            confirmed_points_[i].position - confirmed_points_[j].position);
                    count++;
                }
            }
        }
        float distance_avg = count > 0 ? distance_total / count : 0;

        //在最后一轮对范围和点位不匹配的情况进行再次分配
        if (strategy_->split_count == maxSplitCount - 1) {
            for (auto &item: point_sets_) {
                if (item.range() == 2 && item.range() < item.size()) {
                    selectOptimalPoints(item, image, distance_avg);
                }
            }
        }

        // 返回是否所有点位都已确认
        return isAllPointsConfirmed();

    } catch (const std::exception &e) {
        LOGE(LOG_TAG, "processImage 异常: %s", e.what());
        throw PointException("OpenCV error: " + string(e.what()));
    }
}

void ColorSplitter::processSmallSet(PointSet &set) {
    if (set.points.empty()) return;
    if (set.end_range - set.start_range == 1 && set.size() <= 2) {
//        LOGD(LOG_TAG, "开始推导点位序号 - 点数: %zu, 范围: [%d, %d]",
//             set.points.size(), set.start_range, set.end_range);
        // 优先尝试通过相邻关系推导
        deducePointIds(set);
        // 如果仍未确认，则使用随机分配
        if (!set.points[0].confirmed || !set.points[1].confirmed) {
            assignRandomIds(set);
        }
    } else if (set.end_range - set.start_range == 0 && set.size() == 1) {
//        LOGW(LOG_TAG, "推断仅一个点时 - 点数: %zu, 范围: [%d, %d]",
//             set.points.size(), set.start_range, set.end_range);
        LedPoint &point = set.points[0];
        if (!point.confirmed) {
            point.id = set.start_range;
            point.confirmed = true;
            confirmed_points_.push_back(point);
        }
    }
}

bool ColorSplitter::isAllPointsConfirmed() const {
    // 检查是否有未确认的点位集合
    for (const auto &set: point_sets_) {
        for (const auto &point: set.points) {
            if (!point.confirmed) {
                return false;
            }
        }
    }
    return !confirmed_points_.empty();  // 确保至少有一些确认的点位
}

void ColorSplitter::deducePointIds(PointSet &set) {
    if (set.points.empty()) return;

    if (set.size() == 2) {
        // 对于两点情况，查找前后相邻段最近的点
        Point2f prev_endpoint;  // 前一段最近的点
        Point2f next_endpoint;  // 后一段最近的点
        bool found_prev = false;
        bool found_next = false;

        auto &p1 = set.points[0].position;
        auto &p2 = set.points[1].position;

        // 遍历所有点集找相邻段
        for (const auto &curr_set: point_sets_) {
            if (curr_set.end_range + 1 == set.start_range) {
                // 找到一段，寻找最近的点
                if (!curr_set.points.empty()) {
                    float min_dist = numeric_limits<float>::max();
                    for (const auto &point: curr_set.points) {
                        float dist1 = norm(point.position - p1);
                        float dist2 = norm(point.position - p2);
                        float min_curr_dist = min(dist1, dist2);
                        if (min_curr_dist < min_dist) {
                            min_dist = min_curr_dist;
                            prev_endpoint = point.position;
                            found_prev = true;
                        }
                    }
//                    LOGD(LOG_TAG, "找到前一段最近点: (%.2f, %.2f), 最小距离: %.2f",
//                         prev_endpoint.x, prev_endpoint.y, min_dist);
                }
            } else if (curr_set.start_range == set.end_range + 1) {
                LOGD(LOG_TAG, "找到后一段: %d - %d size=%d",
                     curr_set.start_range, curr_set.end_range, curr_set.points.size());
                // 找到后一段，寻找最近的点
                if (!curr_set.points.empty()) {
                    float min_dist = numeric_limits<float>::max();
                    for (const auto &point: curr_set.points) {
                        float dist1 = norm(point.position - p1);
                        float dist2 = norm(point.position - p2);
                        float min_curr_dist = min(dist1, dist2);
                        if (min_curr_dist < min_dist) {
                            min_dist = min_curr_dist;
                            next_endpoint = point.position;
                            found_next = true;
                        }
                    }
                    LOGD(LOG_TAG, "找到后一段最近点: (%.2f, %.2f), 最小距离: %.2f",
                         next_endpoint.x, next_endpoint.y, min_dist);
                }
            }
        }

        // 根据找到的相邻点确定顺序
        if (found_prev && found_next) {
            // 如果找到前后两段点使用原有逻辑
            float dist_p1_prev = norm(p1 - prev_endpoint);
            float dist_p1_next = norm(p1 - next_endpoint);
            float dist_p2_prev = norm(p2 - prev_endpoint);
            float dist_p2_next = norm(p2 - next_endpoint);

            LOGD(LOG_TAG, "点1(%.2f, %.2f)到前点距离: %.2f, 到后点距离: %.2f",
                 p1.x, p1.y, dist_p1_prev, dist_p1_next);
            LOGD(LOG_TAG, "点2(%.2f, %.2f)到前点距离: %.2f, 到后点距离: %.2f",
                 p2.x, p2.y, dist_p2_prev, dist_p2_next);

            if (dist_p1_prev + dist_p2_next < dist_p2_prev + dist_p1_next) {
                assignIds(set, 0, 1);
            } else {
                assignIds(set, 1, 0);
            }
        } else if (found_prev) {
            // 只找到前一段点，距离前一段点近的是第一个点
            float dist_p1_prev = norm(p1 - prev_endpoint);
            float dist_p2_prev = norm(p2 - prev_endpoint);

            LOGD(LOG_TAG, "仅找到前一段点 - 点1距离: %.2f, 点2距离: %.2f",
                 dist_p1_prev, dist_p2_prev);

            if (dist_p1_prev < dist_p2_prev) {
                assignIds(set, 0, 1);
            } else {
                assignIds(set, 1, 0);
            }
        } else if (found_next) {
            // 只找到后一段点，距后一段点近的是第二个点
            float dist_p1_next = norm(p1 - next_endpoint);
            float dist_p2_next = norm(p2 - next_endpoint);

            LOGD(LOG_TAG, "仅找到后一段点 - 点1距离: %.2f, 点2距离: %.2f",
                 dist_p1_next, dist_p2_next);

            if (dist_p1_next < dist_p2_next) {
                assignIds(set, 1, 0);
            } else {
                assignIds(set, 0, 1);
            }
        } else {
            LOGW(LOG_TAG, "未找到任何相邻段点");
            return;
        }
    }
}

// 新增辅助方法：分配ID并确认点位
void ColorSplitter::assignIds(PointSet &set, int first_index, int second_index) {
    set.points[first_index].id = set.start_range;
    set.points[second_index].id = set.end_range;

    // 确认点位
    for (auto &point: set.points) {
        point.confirmed = true;
        confirmed_points_.push_back(point);
//        LOGD(LOG_TAG, "确认点位 - ID: %d, 位置: (%.2f, %.2f)",
//             point.id, point.position.x, point.position.y);
    }
}

void ColorSplitter::assignRandomIds(PointSet &set) {
    // 创建可分配序号列表
    vector<int> available_ids;
    for (int id = set.start_range; id <= set.end_range; id++) {
        available_ids.push_back(id);
    }

    // 随机打乱序号
    random_device rd;
    mt19937 gen(rd());
    shuffle(available_ids.begin(), available_ids.end(), gen);

    // 分配序号
    size_t id_index = 0;
    for (auto &point: set.points) {
        if (!point.confirmed && id_index < available_ids.size()) {
            point.id = available_ids[id_index++];
            point.confirmed = true;
            confirmed_points_.push_back(point);
        }
    }
}

// 实现 ColorBasedSplitStrategy 的 split 方法
vector<PointSet> ColorBasedSplitStrategy::split(
        const PointSet &input_set,
        const Mat &image
) {
    vector<PointSet> result_sets;
    try {
        LOGW(LOG_TAG, "\n==== 开始第 %d 轮颜色分割 ====", split_count);
        LOGD(LOG_TAG, "输入点集范围: [%d, %d], 点数: %zu",
             input_set.start_range, input_set.end_range, input_set.points.size());

        // 计算点集范围
        int range_size = input_set.end_range - input_set.start_range + 1;
        int sub_range_size = range_size / color_channels_;

        // 创建颜色子集
        vector<PointSet> color_sets(color_channels_);
        for (int i = 0; i < color_channels_; i++) {
            int start = input_set.start_range + i * sub_range_size;
            int end = (i == color_channels_ - 1) ? input_set.end_range : start + sub_range_size - 1;
            color_sets[i] = PointSet(start, end);
//            LOGW(LOG_TAG, "创��颜色子集 %d: 范围[%d, %d] 范围小=%d", i, start, end,
//                 end - start + 1);
        }

        // 用于临时存储分类结果
        vector<vector<pair<LedPoint, pair<float, float>>>> temp_sets(
                2); // {point, {color_ratio, avg_distance}}

        // 第一遍：计算所有点的颜色强度和平均距离
        for (const auto &point: input_set.points) {
            if (point.confirmed) continue;

            Vec3b avg_color = point.calculateAverageColor(image);
            float red_intensity = avg_color[2];
            float green_intensity = avg_color[1];

            // 计算颜色强度比
            float intensity_ratio = max(red_intensity, green_intensity) /
                                    (min(red_intensity, green_intensity) + 1e-6);

            // 计算与其他点的平均距离
            float total_distance = 0;
            int valid_points = 0;
            for (const auto &other_point: input_set.points) {
                if (&point != &other_point && !other_point.confirmed) {
                    total_distance += norm(point.position - other_point.position);
                    valid_points++;
                }
            }
            float avg_distance = valid_points > 0 ? total_distance / valid_points : 0;

            int dominant_channel = (red_intensity > green_intensity) ? 0 : 1;
            temp_sets[dominant_channel].push_back({point, {intensity_ratio, avg_distance}});
        }

        LOGD(LOG_TAG, "[第%d轮] 初步分类 - 红色组: %zu点, 绿色组: %zu点",
             split_count, temp_sets[0].size(), temp_sets[1].size());

        // 第二遍：处理离群点和超出范围的情况
        for (int i = 0; i < 2; i++) {
            if (temp_sets[i].empty()) continue;

            // 计算距离的均值和标准差
            float sum_distance = 0;
            for (const auto &pair: temp_sets[i]) {
                sum_distance += pair.second.second;
            }
            float mean_distance = sum_distance / temp_sets[i].size();

            float sum_squared_diff = 0;
            for (const auto &pair: temp_sets[i]) {
                float diff = pair.second.second - mean_distance;
                sum_squared_diff += diff * diff;
            }
            float std_dev = sqrt(sum_squared_diff / temp_sets[i].size());
            float distance_threshold = mean_distance + 3 * std_dev;
            if (split_count == 0) {
                distance_threshold = mean_distance + 6 * std_dev;
            } else if (split_count == 1) {
                distance_threshold = mean_distance + 5 * std_dev;
            } else if (split_count == 2) {
                distance_threshold = mean_distance + 4 * std_dev;
            }
//            LOGD(LOG_TAG, "[第%d轮] %s组 - 平均距离:%.2f, 标准差:%.2f, 阈值:%.2f",
//                 split_count, i == 0 ? "红色" : "绿色",
//                 mean_distance, std_dev, distance_threshold);

            // 标记需要移除的点
            vector<size_t> points_to_remove;
            for (size_t j = 0; j < temp_sets[i].size(); j++) {
                const auto &pair = temp_sets[i][j];
                if (pair.second.second > distance_threshold) {
                    // 检查是否可以移动到另一组
                    bool can_move_to_other = true;
                    if (!temp_sets[1 - i].empty()) {
                        // 计算与另一组点的平均距离
                        float total_other_distance = 0;
                        for (const auto &other_pair: temp_sets[1 - i]) {
                            total_other_distance += norm(
                                    pair.first.position - other_pair.first.position);
                        }
                        float avg_other_distance = total_other_distance / temp_sets[1 - i].size();

                        // 如果在另一组也是离群点，或者颜色比过高，则标记为删除
                        if (avg_other_distance > distance_threshold || pair.second.first > 1.2) {
                            can_move_to_other = false;
                        }
                    }

                    if (can_move_to_other) {
                        // 移动到另一组
                        LOGW(LOG_TAG, "(%.2f, %.2f)从%s组移动到%s组",
                             pair.first.position.x, pair.first.position.y,
                             i == 0 ? "红色" : "绿色",
                             i == 0 ? "绿色" : "红色");
                        temp_sets[1 - i].push_back(
                                {pair.first, {1.0f / pair.second.first, pair.second.second}});
                        points_to_remove.push_back(j);
                    } else {
                        // 标记为删除
                        points_to_remove.push_back(j);
                        LOGE(LOG_TAG, "离群点(%.2f, %.2f)被删除 - 距离:%.2f, 颜色比:%.2f",
                             pair.first.position.x, pair.first.position.y,
                             pair.second.second, pair.second.first);
                    }
                }
            }

            // 从后向前删除标的点
            sort(points_to_remove.begin(), points_to_remove.end(), greater<size_t>());
            for (size_t idx: points_to_remove) {
                temp_sets[i].erase(temp_sets[i].begin() + idx);
            }
        }

        // 最终分配点位
        for (int i = 0; i < 2; i++) {
            for (const auto &pair: temp_sets[i]) {
                color_sets[i].points.push_back(pair.first);
            }

            if (!color_sets[i].points.empty()) {
                color_sets[i].colorType = i;
                result_sets.push_back(color_sets[i]);
                LOGD(LOG_TAG, "[第%d轮] 最终%s组 - 范围[%d, %d], 点数: %zu",
                     split_count, i == 0 ? "红色" : "绿色",
                     color_sets[i].start_range, color_sets[i].end_range,
                     color_sets[i].points.size());
            }
        }

    } catch (const exception &e) {
        LOGE(LOG_TAG, "颜色分割错误: %s", e.what());
        throw PointException("Error in color-based split: " + string(e.what()));
    }

    return result_sets;
}

vector<vector<int>> ColorSplitter::getSplitColorMapping(int split_level) const {
    vector<vector<int>> result;
    // 验证分割次数的有效性
    if (split_level < 0 || split_level > getMaxSplitCount()) {
        LOGE(LOG_TAG, "Invalid split level: %d", split_level);
        return result;
    }

    // 检查缓存
    LOGD(LOG_TAG, "使用缓存优化 color_mapping_cache_ : %d", color_mapping_cache_.size());
    auto it = color_mapping_cache_.find(split_level);
    if (it != color_mapping_cache_.end()) {
        return it->second;
    } else {
        LOGE(LOG_TAG, "Invalid color mapping");
    }

    return result;
}

void ColorSplitter::selectOptimalPoints(PointSet &set, const Mat &image, float distanceAvg) {
    if (set.points.empty() || confirmed_points_.empty()) return;

    LOGD(LOG_TAG, "开始选择最优点 - 点数: %zu, 范围: [%d, %d]",
         set.points.size(), set.start_range, set.end_range);

    // 找到与当前范围相邻的已确认点
    vector<LedPoint> nearby_confirmed;
    int loopTimes = 0;
    int cnt = 3;
    while (nearby_confirmed.size() < 2 && loopTimes < 3) {
        nearby_confirmed.clear();
        for (const auto &confirmed: confirmed_points_) {
            if (abs(confirmed.id - set.start_range) <= cnt ||
                abs(confirmed.id - set.end_range) <= cnt) {
                nearby_confirmed.push_back(confirmed);
            }
        }
        cnt = cnt + 5;
        loopTimes++;
    }

    if (nearby_confirmed.size() < 2) {
        LOGW(LOG_TAG, "nearby_confirmed = %d", nearby_confirmed.size());
        return;
    }

    LOGD(LOG_TAG, "距离统计 - 平均distanceAvg: %.2f,  临近点数量：%d", distanceAvg,
         nearby_confirmed.size());

    // 为每个候选点计算得分
    vector<pair<LedPoint *, float>> point_scores;
    for (auto &point: set.points) {
        if (point.confirmed) continue;

        float score = 0;
        float dist_score = 0;
        bool is_valid = true;

        // 1. 检查与相邻已确认点的距离
        for (const auto &nearby: nearby_confirmed) {
            float dist = norm(point.position - nearby.position);
            int id_diff = abs(nearby.id - set.start_range);

            // 期望距离应该与ID差值成正比
            float expected_dist = distanceAvg * id_diff;

            // 如果距离偏差太大，标记为无效点
            if (abs(dist) > abs(expected_dist) * 3) {
//                LOGW(LOG_TAG, "点 (%.1f, %.1f) 距离偏差过大: %.2f > %.2f",
//                     point.position.x, point.position.y,
//                     abs(dist), abs(expected_dist) * 3);
                is_valid = false;
                break;
            }

            dist_score += -dist / id_diff;
        }

        if (!is_valid) continue;  // 跳过无效点

        score += dist_score;

        point_scores.push_back({&point, score});

        LOGD(LOG_TAG, "点 (%.1f, %.1f) 得分: %.2f (距离: %.2f, 颜色: %.2f)",
             point.position.x, point.position.y, score,
             dist_score);
    }

    // 按得分排序
    sort(point_scores.begin(), point_scores.end(),
         [](const auto &a, const auto &b) { return a.second > b.second; });

    // 选择得分最高的两个点
    for (size_t i = 0; i < min(size_t(2), point_scores.size()); i++) {
        auto *point = point_scores[i].first;
        point->id = set.start_range + i;
        point->confirmed = true;
        confirmed_points_.push_back(*point);

        LOGD(LOG_TAG, "选择点 (%.1f, %.1f) 作为序号 %d, 得分: %.2f",
             point->position.x, point->position.y, point->id, point_scores[i].second);
    }

    // 移除未选中的点
    set.points.erase(
            remove_if(set.points.begin(), set.points.end(),
                      [](const LedPoint &p) { return !p.confirmed; }),
            set.points.end()
    );

    LOGD(LOG_TAG, "完成最优点选择 - 剩余点数: %zu", set.points.size());
}

from metrics.simple_statistics_metric import SimpleStatisticsMetric


class ForwardTimeMetric(SimpleStatisticsMetric):
    def add_forward_times(self, needed_times):
        self.add_statistics_entries(needed_times)

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])

report.run(reference_data=reference_data, current_data=current_data)
report.save_html('report.html')

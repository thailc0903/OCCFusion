"""Task dispatch defaults preserve legacy occupancy profile JSON/fingerprints."""
OCCUPANCY='occupancy'
DETECTION='detection'
DETECTION_FILES=('detections.json','metadata.json','report.json','resolved_config.py')

def task_of(profile):
    value=profile.get('task',OCCUPANCY)
    if value not in (OCCUPANCY,DETECTION):raise ValueError('Unsupported inference task')
    return value

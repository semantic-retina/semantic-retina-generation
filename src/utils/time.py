def format_seconds(total_seconds: int):
    """Formats elapsed seconds into hours, minutes, and seconds."""
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    delta_hours = f"{hours:02d}"
    delta_minutes = f"{minutes:02d}"
    delta_seconds = f"{seconds:02d}"
    return f"{delta_hours}:{delta_minutes}:{delta_seconds}"

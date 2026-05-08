"""
Event-Based Trade Filter
Filters trades based on economic events (CPI, FOMC, NFP, etc.)
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger()


class EventFilter:
    """
    Filters trading signals based on economic events
    Pauses trading before/after major events to avoid volatility
    """

    def __init__(self, config: Dict):
        self.config = config
        self.events_config = config.get('events', {})

        self.enabled = self.events_config.get('enabled', True)
        self.pause_before_hours = self.events_config.get(
            'pause_before_event_hours', 2)
        self.pause_after_hours = self.events_config.get(
            'pause_after_event_hours', 1)
        self.major_events = self.events_config.get('major_events', [
            'FOMC', 'CPI', 'NFP', 'GDP', 'Interest Rate Decision'
        ])

        # Upcoming events (manually added for now)
        self.upcoming_events = []

        logger.info(f"Event filter initialized - Enabled: {self.enabled}")

    def should_trade(self, timestamp: datetime = None) -> bool:
        """
        Check if trading is allowed at the given time

        Args:
            timestamp: Time to check (defaults to now)

        Returns:
            True if trading allowed, False if paused for event
        """
        if not self.enabled:
            return True

        if timestamp is None:
            timestamp = datetime.now()

        # Check if we're near any major events
        for event in self.upcoming_events:
            event_time = event.get('time')
            if event_time is None:
                continue

            pause_start = event_time - timedelta(hours=self.pause_before_hours)
            pause_end = event_time + timedelta(hours=self.pause_after_hours)

            if pause_start <= timestamp <= pause_end:
                logger.warning(
                    f"Trading paused due to {event.get('name')} at {event_time}")
                return False

        return True

    def add_event(self, event_name: str, event_time: datetime, impact: str = 'high'):
        """
        Add an upcoming economic event

        Args:
            event_name: Name of the event (e.g., 'FOMC Meeting')
            event_time: When the event occurs
            impact: Expected impact level (low/medium/high)
        """
        # Only track major events
        if not any(major in event_name for major in self.major_events):
            return

        self.upcoming_events.append({
            'name': event_name,
            'time': event_time,
            'impact': impact
        })

        logger.info(f"Added economic event: {event_name} at {event_time}")

    def clear_past_events(self, current_time: datetime = None):
        """
        Remove events that have already passed

        Args:
            current_time: Current time (defaults to now)
        """
        if current_time is None:
            current_time = datetime.now()

        # Keep only future events (with some buffer)
        cutoff_time = current_time - timedelta(hours=self.pause_after_hours)
        self.upcoming_events = [
            event for event in self.upcoming_events
            if event.get('time') and event['time'] > cutoff_time
        ]

    def get_upcoming_events(self, hours_ahead: int = 24) -> List[Dict]:
        """
        Get events happening in the next N hours

        Args:
            hours_ahead: How many hours to look ahead

        Returns:
            List of upcoming events
        """
        current_time = datetime.now()
        cutoff_time = current_time + timedelta(hours=hours_ahead)

        return [
            event for event in self.upcoming_events
            if event.get('time') and current_time <= event['time'] <= cutoff_time
        ]

    def get_next_event(self) -> Optional[Dict]:
        """
        Get the next upcoming event

        Returns:
            Next event dictionary or None
        """
        current_time = datetime.now()

        future_events = [
            event for event in self.upcoming_events
            if event.get('time') and event['time'] > current_time
        ]

        if not future_events:
            return None

        # Sort by time and return earliest
        return min(future_events, key=lambda x: x['time'])

    def get_pause_status(self) -> Dict:
        """
        Get current pause status and reason

        Returns:
            Dictionary with pause status information
        """
        current_time = datetime.now()

        for event in self.upcoming_events:
            event_time = event.get('time')
            if event_time is None:
                continue

            pause_start = event_time - timedelta(hours=self.pause_before_hours)
            pause_end = event_time + timedelta(hours=self.pause_after_hours)

            if pause_start <= current_time <= pause_end:
                if current_time < event_time:
                    status = "paused_before"
                    time_until = event_time - current_time
                else:
                    status = "paused_after"
                    time_until = pause_end - current_time

                return {
                    'paused': True,
                    'status': status,
                    'event': event.get('name'),
                    'event_time': event_time,
                    'resume_time': pause_end,
                    'time_remaining': time_until
                }

        return {
            'paused': False,
            'status': 'active',
            'event': None,
            'event_time': None,
            'resume_time': None,
            'time_remaining': None
        }

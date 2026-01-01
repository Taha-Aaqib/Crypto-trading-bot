# Economic calendar fetcher (CPI, FOMC)


class EconomicCalendar:
    """
    Fetch economic events from free APIs

    Sources:
    1. Trading Economics API (free tier)
    2. Forex Factory (web scraping as fallback)
    3. Manual event file (config/economic_events.json)
    """

    def __init__(self):
        """Initialize economic calendar fetcher"""
        self.cache_file = "data/economic_events_cache.json"
        self.manual_events_file = "config/economic_events.json"
        self.cache_duration_hours = 24  # Cache events for 24 hours

        logger.info("Economic calendar initialized")

    def fetch_upcoming_events(
        self,
        days_ahead: int = 7,
        importance: str = "high"
    ) -> List[Dict]:
        """
        Fetch upcoming economic events

        Args:
            days_ahead: Number of days to look ahead
            importance: Event importance filter ('high', 'medium', 'low')

        Returns:
            List of event dictionaries with:
            - name: Event name
            - country: Country code
            - date: Event datetime
            - importance: high/medium/low
            - actual: Actual value (if released)
            - forecast: Forecasted value
            - previous: Previous value
        """
        logger.info(f"Fetching economic events for next {days_ahead} days...")

        # Try to load from cache first
        cached_events = self._load_from_cache()
        if cached_events:
            logger.info(f"Loaded {len(cached_events)} events from cache")
            return self._filter_events(cached_events, days_ahead, importance)

        # Try to fetch from free API
        events = self._fetch_from_trading_economics(days_ahead)

        # If API fails, fall back to manual events file
        if not events:
            logger.warning("API fetch failed, loading manual events")
            events = self._load_manual_events()

        # Save to cache
        if events:
            self._save_to_cache(events)

        # Filter by importance and time range
        filtered_events = self._filter_events(events, days_ahead, importance)

        logger.info(
            f"Found {len(filtered_events)} upcoming {importance}-importance events")

        return filtered_events

    def _fetch_from_trading_economics(self, days_ahead: int) -> List[Dict]:
        """
        Fetch from Trading Economics free calendar
        Note: Free tier has limited access
        """
        try:
            # Trading Economics public calendar (no API key needed for basic access)
            url = "https://tradingeconomics.com/calendar"

            # For demo/FYP purposes, use manual events
            # In production, you would implement proper API integration
            logger.info(
                "Trading Economics API requires subscription - using manual events")
            return []

        except Exception as e:
            logger.error(f"Error fetching from Trading Economics: {e}")
            return []

    def _load_manual_events(self) -> List[Dict]:
        """
        Load manually configured events from JSON file
        This serves as fallback and allows students to add custom events
        """
        if not os.path.exists(self.manual_events_file):
            logger.warning(
                f"Manual events file not found: {self.manual_events_file}")
            return self._create_default_events()

        try:
            with open(self.manual_events_file, 'r') as f:
                events = json.load(f)

            logger.info(f"Loaded {len(events)} manual events")
            return events

        except Exception as e:
            logger.error(f"Error loading manual events: {e}")
            return self._create_default_events()

    def _create_default_events(self) -> List[Dict]:
        """
        Create default major economic events for current year
        Based on typical schedule (FOMC 8x/year, CPI monthly, NFP monthly)
        """
        logger.info("Creating default economic events")

        current_year = datetime.now().year
        events = []

        # FOMC Meetings (typically 8 per year, roughly every 6 weeks)
        # Typical months: Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
        fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
        for month in fomc_months:
            # Usually second or third Wednesday
            event_date = datetime(current_year, month, 15, 14, 0)  # 2 PM ET
            if event_date > datetime.now():
                events.append({
                    "name": "FOMC Meeting",
                    "country": "USD",
                    "date": event_date.isoformat(),
                    "importance": "high",
                    "description": "Federal Reserve interest rate decision"
                })

        # CPI (Consumer Price Index) - Monthly, typically around 13th-15th
        for month in range(1, 13):
            event_date = datetime(current_year, month, 13, 8, 30)  # 8:30 AM ET
            if event_date > datetime.now():
                events.append({
                    "name": "CPI",
                    "country": "USD",
                    "date": event_date.isoformat(),
                    "importance": "high",
                    "description": "Consumer Price Index (inflation data)"
                })

        # NFP (Non-Farm Payrolls) - First Friday of each month
        for month in range(1, 13):
            # Find first Friday
            first_day = datetime(current_year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            event_date = first_day + timedelta(days=days_until_friday)
            event_date = event_date.replace(hour=8, minute=30)  # 8:30 AM ET

            if event_date > datetime.now():
                events.append({
                    "name": "NFP",
                    "country": "USD",
                    "date": event_date.isoformat(),
                    "importance": "high",
                    "description": "Non-Farm Payrolls (employment data)"
                })

        # GDP (Quarterly)
        gdp_months = [1, 4, 7, 10]  # End of each quarter
        for month in gdp_months:
            event_date = datetime(current_year, month, 28, 8, 30)
            if event_date > datetime.now():
                events.append({
                    "name": "GDP",
                    "country": "USD",
                    "date": event_date.isoformat(),
                    "importance": "high",
                    "description": "Gross Domestic Product"
                })

        logger.info(f"Created {len(events)} default events")

        # Save to manual events file for future use
        os.makedirs(os.path.dirname(self.manual_events_file), exist_ok=True)
        with open(self.manual_events_file, 'w') as f:
            json.dump(events, f, indent=2)

        return events

    def _filter_events(
        self,
        events: List[Dict],
        days_ahead: int,
        importance: str
    ) -> List[Dict]:
        """Filter events by time range and importance"""
        now = datetime.now()
        future_cutoff = now + timedelta(days=days_ahead)

        filtered = []
        for event in events:
            # Parse event date
            try:
                event_date = pd.to_datetime(event['date'])
            except:
                continue

            # Check if within time range
            if now <= event_date <= future_cutoff:
                # Check importance
                event_importance = event.get('importance', 'medium').lower()

                if importance == 'high' and event_importance == 'high':
                    filtered.append(event)
                elif importance == 'medium' and event_importance in ['high', 'medium']:
                    filtered.append(event)
                elif importance == 'low':
                    filtered.append(event)

        return filtered

    def _load_from_cache(self) -> Optional[List[Dict]]:
        """Load events from cache if not expired"""
        if not os.path.exists(self.cache_file):
            return None

        try:
            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(self.cache_file)
            )

            if cache_age.total_seconds() > self.cache_duration_hours * 3600:
                logger.info("Cache expired, fetching fresh data")
                return None

            with open(self.cache_file, 'r') as f:
                events = json.load(f)

            return events

        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None

    def _save_to_cache(self, events: List[Dict]):
        """Save events to cache file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

            with open(self.cache_file, 'w') as f:
                json.dump(events, f, indent=2)

            logger.info(f"Cached {len(events)} events")

        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def is_major_event_near(
        self,
        hours_before: int = 2,
        hours_after: int = 1
    ) -> Dict:
        """
        Check if a major economic event is near current time

        Args:
            hours_before: Hours before event to avoid trading
            hours_after: Hours after event to avoid trading

        Returns:
            Dictionary with:
            - event_near: Boolean
            - event: Event details if near, else None
            - time_until: Hours until event (negative if passed)
        """
        events = self.fetch_upcoming_events(days_ahead=1, importance='high')

        now = datetime.now()

        for event in events:
            event_time = pd.to_datetime(event['date'])
            time_diff = (event_time - now).total_seconds() / 3600  # Hours

            # Check if within avoidance window
            if -hours_after <= time_diff <= hours_before:
                logger.warning(
                    f"Major event near: {event['name']} in {time_diff:.1f} hours")
                return {
                    'event_near': True,
                    'event': event,
                    'time_until': time_diff
                }

        return {
            'event_near': False,
            'event': None,
            'time_until': None
        }

    def get_events_this_week(self) -> pd.DataFrame:
        """
        Get all events for current week as DataFrame
        Useful for dashboard display
        """
        events = self.fetch_upcoming_events(days_ahead=7, importance='low')

        if not events:
            return pd.DataFrame()

        df = pd.DataFrame(events)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        return df

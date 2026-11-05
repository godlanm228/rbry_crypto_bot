"""The background series: DeFi's locked value and the Fed's decision days (plan item 8)."""

from core.sandbox import context_sources as cs

FED = '''<h4><a>2025 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
<div class="fomc-meeting__date col-xs-4">28-29</div></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>Apr/May</strong></div>
<div class="fomc-meeting__date col-xs-4">30-1</div></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>March</strong></div>
<div class="fomc-meeting__date col-xs-4">18-19*</div></div>'''


def test_the_fed_decides_on_a_meetings_last_day_even_across_a_month():
    assert cs.parse_fomc(FED) == [("2025-01-29", 1.0), ("2025-03-19", 1.0), ("2025-05-01", 1.0)]


def test_tvl_by_utc_day_without_the_empty_start():
    rows = [{"date": 1506470400, "tvl": 0}, {"date": 1518393600, "tvl": 6.6e6}]
    assert cs.parse_tvl(rows) == [("2018-02-12", 6.6e6)]


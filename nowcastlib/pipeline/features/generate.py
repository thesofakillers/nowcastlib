"""functions for generating new fields"""
import numpy as np
import pandas as pd
from skyfield import api as skyfield_api
from skyfield import almanac
from nowcastlib.pipeline import structs


# {{{ sec_day
def _trig_sec(datetime_series, trig_func):
    """takes the sine or cos of the current second of the day"""
    seconds_in_day = 24 * 60 * 60
    secs_since_midnight = (
        datetime_series - datetime_series.dt.normalize()
    ).total_seconds()
    return trig_func((2 * np.pi * secs_since_midnight) / seconds_in_day)


def _sin_sec(datetime_series):
    """takes the sine of the current second of the day"""
    return _trig_sec(datetime_series, np.sin)


def _cos_sec(datetime_series):
    """takes the cosine of the current second of the day"""
    return _trig_sec(datetime_series, np.cos)


# }}}


# {{{ day_year
def _trig_day_year(datetime_series, trig_func):
    """takes the sine or cosine of the current day of the year"""
    days_in_year = np.where(datetime_series.dt.is_leap_year, 366, 365)
    days_since_newyears = datetime_series.dt.day_of_year
    return trig_func((2 * np.pi * days_since_newyears) / days_in_year)


def _sin_day_year(datetime_series):
    """takes the sine of the current day of the year"""
    return _trig_day_year(datetime_series, np.sin)


def _cos_day_year(datetime_series):
    """takes the cosine of the current day of the year"""
    return _trig_day_year(datetime_series, np.cos)


# }}}

# {{{ day_week
def _trig_day_week(datetime_series, trig_func):
    """takes the sine or cosine of the current day number of the week (7)"""
    days_in_week = 7
    day_of_week = datetime_series.dt.day_of_week + 1
    return trig_func((2 * np.pi * day_of_week) / days_in_week)


def _sin_day_week(datetime_series):
    """takes the sine of the current day number of the week (7)"""
    return _trig_day_week(datetime_series, np.sin)


def _cos_day_week(datetime_series):
    """takes the cosine of the current day number of the week (7)"""
    return _trig_day_week(datetime_series, np.cos)


# }}}

# {{{ month_year
def _trig_month_year(datetime_series, trig_func):
    """takes the sine or cosine of the current month number of the year (12)"""
    months_in_year = 12
    month_of_year = datetime_series.dt.month
    return trig_func((2 * np.pi * month_of_year) / months_in_year)


def _sin_month_year(datetime_series):
    """takes the sine of the current month number of the year (12)"""
    return _trig_month_year(datetime_series, np.sin)


def _cos_month_year(datetime_series):
    """takes the cosine of the current month number of the year (12)"""
    return _trig_month_year(datetime_series, np.cos)


# }}}


def _is_weekend(datetime_series):
    """returns a boolean series of 1 if weekend and 0 otherwise"""
    return (datetime_series.dt.day_of_week >= 4).astype(int)


# {{{ time since last sunset


def _t_since_sunset(
    datetime_series,
    lat=-24.6275,
    lon=-70.4044,
    elevation=2635,
):
    """returns how many seconds have elapsed since sunset"""
    # loading skyfield objects
    eph = skyfield_api.load("de421.bsp")
    skyfield_ts = skyfield_api.load.timescale()
    location = skyfield_api.wgs84.latlon(lat, lon, elevation_m=elevation)
    # start and end time in UTC in python datetime format
    start_ts = skyfield_ts.from_datetime(
        # minus one day because we are interested in _previous_ sunset
        (datetime_series[0] - pd.to_timedelta(1, unit="d"))
        .tz_localize("UTC")
        .to_pydatetime()
    )
    end_ts = skyfield_ts.from_datetime(
        datetime_series[-1].tz_localize("UTC").to_pydatetime()
    )
    # find when sunsets occurred between our start and end dates
    times, rise_set_mask = almanac.find_discrete(
        start_ts, end_ts, almanac.sunrise_sunset(eph, location)
    )
    sunsets = pd.to_datetime(times.utc_iso())[
        ~np.array(rise_set_mask).astype(bool)
    ].tz_localize(None)
    # find when the most recent sunset was for each timestep in input datetime series
    sunset_idxs = np.zeros(len(datetime_series))
    for sunset in sunsets[1:]:
        change = np.where(datetime_series > sunset)[0][0]
        sunset_idxs[change:] += 1
    prev_sunset = sunsets[sunset_idxs]
    # find how many seconds since most recent sunset for each timestep
    return (datetime_series - prev_sunset).dt.total_seconds()


def _trig_t_since_sunset(
    datetime_series, trig_func, lat=-24.6275, lon=-70.4044, elevation=2635
):
    """
    returns the sine or cosine of how many seconds have
    elapsed since sunset out of 86400
    """
    seconds_in_day = 60 * 60 * 24
    return trig_func(
        2
        * np.pi
        * _t_since_sunset(datetime_series, lat, lon, elevation)
        / seconds_in_day
    )


def _sin_t_since_sunset(datetime_series, lat=-24.6275, lon=-70.4044, elevation=2635):
    """
    returns the cosine of how many seconds have
    elapsed since sunset out of 86400
    """
    return _trig_t_since_sunset(datetime_series, np.sin, lat, lon, elevation)


def _cos_t_since_sunset(datetime_series, lat=-24.6275, lon=-70.4044, elevation=2635):
    """
    returns the sine of how many seconds have
    elapsed since sunset out of 86400
    """
    return _trig_t_since_sunset(datetime_series, np.cos, lat, lon, elevation)


# }}}


def _sun_elevation(datetime_series, lat=-24.6275, lon=-70.4044, elevation=2635):
    """returns the sun elevation in degrees at each timestamp for a given location"""
    # load skyfield reqs
    eph = skyfield_api.load("de421.bsp")
    skyfield_ts = skyfield_api.load.timescale()
    # specify location is relative to earth in the context of solar system barycentre
    location = eph["earth"] + skyfield_api.wgs84.latlon(lat, lon, elevation_m=elevation)
    # get the astrometric and apparent positions of the sun from this location
    astrometric_pos = location.at(
        skyfield_ts.from_datetimes(
            datetime_series.dt.tz_localize("UTC").to_pydatetime()
        )
    ).observe(eph["sun"])
    apparent_pos = astrometric_pos.apparent()
    # finally can calculate the altitude (aka the elevation), along with other metrics
    alt, _, _ = apparent_pos.altaz()
    return alt.degrees


function_map: dict = {
    structs.GeneratorFunction.SUN_ELEVATION: _sun_elevation,
    structs.GeneratorFunction.T_SINCE_SUNSET: _t_since_sunset,
    structs.GeneratorFunction.SIN_T_SINCE_SUNSET: _sin_t_since_sunset,
    structs.GeneratorFunction.COS_T_SINCE_SUNSET: _cos_t_since_sunset,
    structs.GeneratorFunction.SIN_SEC: _sin_sec,
    structs.GeneratorFunction.COS_SEC: _cos_sec,
    structs.GeneratorFunction.SIN_DAY_YEAR: _sin_day_year,
    structs.GeneratorFunction.COS_DAY_YEAR: _cos_day_year,
    structs.GeneratorFunction.SIN_DAY_WEEK: _sin_day_week,
    structs.GeneratorFunction.COS_DAY_WEEK: _cos_day_week,
    structs.GeneratorFunction.SIN_MONTH_YEAR: _sin_month_year,
    structs.GeneratorFunction.COS_MONTH_YEAR: _cos_month_year,
    structs.GeneratorFunction.IS_WEEKEND: _is_weekend,
}

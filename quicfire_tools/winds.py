"""
QUIC-Fire Tools Wind Module
"""
from __future__ import annotations
import csv
from pathlib import Path

# External Imports
from pydantic import BaseModel, PositiveInt, PositiveFloat, NonNegativeFloat, NonNegativeInt, computed_field
from typing import Union

class WindSensor(BaseModel):
    """
    Class containing information on windshifts for one sensor
    """
    sensor_number: PositiveInt = 1
    time_now: NonNegativeInt
    wind_times: Union[NonNegativeFloat,list(NonNegativeFloat)]
    wind_speeds: Union[PositiveFloat, list(PositiveFloat)]
    wind_directions: Union[PositiveFloat, list(PositiveFloat)]
    sensor_height: PositiveFloat
    x_location: PositiveInt
    y_location: PositiveInt

    @computed_field
    @property
    def _sensor_name(sensor_number):
        return("sensor"+sensor_number)
    
    def _validate_wind_lists(self):
        if isinstance(self.wind_times,float): self.wind_times = [self.wind_times]
        if isinstance(self.wind_speeds, float): self.wind_speeds = [self.wind_speeds]
        if isinstance(self.wind_directions, float): self.wind_directions = [self.wind_directions]
        if len(self.wind_times) != len(self.wind_speeds) != len(self.wind_directions):
            raise ValueError(f"WindSensor: lists of wind times, speeds, and directions must be the same length.\n",
                             f"len(wind_times) = {len(self.wind_times)}\n",
                             f"len(wind_speeds) = {len(self.wind_speeds)}\n",
                             f"len(win_directions) = {len(self.wind_directions)}")
    
    def __str__(self):
        self._validate_wind_lists()
        location_lines = (f"{self.x_location} !X coordinate (meters)\n",
                          f"{self.y_location} !Y coordinate (meters)\n")
        windshifts = []
        for i in len(self.wind_times):
            shift = (f"\n{self.time_now + self.wind_times[i]} !Begining of time step in Unix Epoch time (integer seconds since 1970/1/1 00:00:00)\n",
                     f"1 !site boundary layer flag (1 = log, 2 = exp, 3 = urban canopy, 4 = discrete data points)\n",
                     f"0.1 !site zo\n",
                     f"0. ! 1/L (default = 0)\n"
                     f"!Height (m),Speed	(m/s), Direction (deg relative to true N)\n"
                     f"{self.sensor_height} {self.wind_speeds[i]} {self.wind_directions[i]}")
            windshifts.append(shift)
        
        return location_lines + "".join

    
    @classmethod
    def from_csv(cls,
                 filename: Union[str,Path]):
        """
        Create windshifts from a .csv file.
        
        """
        if isinstance(filename, str):
            filename = Path(filename)
       
        with open(filename,"r") as f:
            matrix = list(csv.reader(f))
        
        # TODO: add function for calculating windspeed/direction from U,V winds. Put in quicfire_tools.utils
class QUIC_fire(InputFile):
    def __init__(self,
                 nx: int,
                 ny: int,
                 nz: int,
                 output_time: int,
                 time_now: int, #WHERE IS THIS CALCULATED
                 sim_time: int, #HOW TO DEAL WITH SIM PARAMS
                 fire_flag: int = 1,
                 random_seed: int = 47,
                 fire_time_step: int = 1,
                 quic_time_step: int = 1,
                 stretch_grid_flag: int = 0,
                 file_path: str = "",
                 dz_array: list[float] = None,
                 fuel_flag: int = 3,
                 fuel_params: list[float] = None,
                 ignition_flag: int = 7,
                 ignition_params: list[int] = None,
                 ignitions_per_cell: int = 2,
                 firebrand_flag: int = 0,
                 auto_kill: int = 1,
                 # Output flags
                 eng_to_atm_out: int = 1,
                 react_rate_out: int = 0,
                 fuel_dens_out: int = 1,
                 QF_wind_out: int = 1,
                 QU_wind_inst_out: int = 1,
                 QU_wind_avg_out: int = 0,
                 fuel_moist_out: int = 1,
                 mass_burnt_out: int = 1,
                 firebrand_out: int = 0,
                 emissions_out: int = 0,
                 radiation_out: int = 0,
                 intensity_out: int = 0):
        """
        Initialize the QU_Simparams class to manage simulation parameters.

        Parameters
        ----------
        nx : int
            Number of cells in the x-direction.
        ny : int
            Number of cells in the y-direction.
        nz : int
            Number of fire grid cells in the z-direction.
        output_time : int
            After how many time steps to print out:
                - fire-related files (excluding emissions and radiation)
                - average emissions and radiation
            After how many quic updates to print out:
                - wind-related files
                - averaged wind-related files
            Use -1 to provide custom times in file QFire_ListOutputTimes.inp
        time_now : int
            When the fire is ignited in Unix Epoch time (integer seconds since 1970/1/1 00:00:00). Must be greater or equal to the time of the first wind
        sim_time : int
            Total simulation time for the fire [s]
        fire_flag : int
            Fire flag, 1 = run fire; 0 = no fire
        random_seed : int
            Random number generator, -1: use time and date, any other integer > 0 is used as the seed
        fire_time_step : int
            time step for the fire simulation [s]
        quic_time_step : int
            Number of fire time steps done before updating the quic wind field (integer, >= 1)
        stretch_grid_flag : int
            Vertical stretching flag: 0 = uniform dz, 1 = custom
        file_path : str
            Path to files defining fuels, ignitions, and topography, with file separator at the end. Defaults to "", indicating files are in the same directory as all other input files
        dz_array : list[float]
            custom dz, one dz per line must be specified, from the ground to the top of the domain
         fuel_flag : int
            Flag for fuel inputs:
                - density
                - moisture
                - height
            1 = uniform; 2 = provided thru QF_FuelDensity.inp, 3 = Firetech files for quic grid, 4 = Firetech files for different grid (need interpolation)
        fuel_params : list[float]
            List of fuel parameters for a uniform grid (fuel_flag = 1) in the order [density, moisture, height]. All must be real numbers 0-1
        ignition_flag : int
            1 = rectangle, 2 = square ring, 3 = circular ring, 4 = file (QF_Ignitions.inp), 5 = time-dependent ignitions (QF_IgnitionPattern.inp), 7 = ignite.dat (firetech)
        ignition_params: list[int]
            List of ignitions parameters to define locations for rectangle, square ring, and circular ring ignitions.
            For all ignition patterns, the following four parameters must be provided in order:
                - Southwest corner in the x-direction (m)
                - Southwest corner in the y-direction(m)
                - Length in the x-direction (m)
                - Length in the y-direction (m)             
            Additional paramters only for square ring pattern (ignition_flag = 2):
                - Width of the ring in the x-direction (m)
                - Width of the ring in the y-direction (m)
            Additional paramters only for circular ring pattern (ignition_flag = 3):
                - Width of the ring (m)
        ignitions_per_cell: int
            Number of ignition per cell of the fire model. Recommended max value of 100
        firebrand_flag : int
            Firebrand flag, 0 = off; 1 = on
            Recommended value = 0 ; firebrands are untested for small scale problems
        auto_kill : int
            Kill if the fire is out and there are no more ignitions or firebrands (0 = no, 1 = yes)
        eng_to_atm_out : int
            Output flag [0, 1]: gridded energy-to-atmosphere (3D fire grid + extra layers)
        react_rate_out : int
            Output flag [0, 1]: compressed array reaction rate (fire grid)
        fuel_dens_out : int
            Output flag [0, 1]: compressed array fuel density (fire grid)
        QF_wind_out : int
            Output flag [0, 1]: gridded wind (u,v,w,sigma) (3D fire grid)
        QU_wind_inst_out : int
            Output flag [0, 1]: gridded QU winds with fire effects, instantaneous (QUIC-URB grid)
        QU_wind_avg_out : int
            Output flag [0, 1]: gridded QU winds with fire effects, averaged (QUIC-URB grid)
        fuel_moist_out : int
            Output flag [0, 1]: compressed array fuel moisture (fire grid)
        mass_burnt_out : int
            Output flag [0, 1]: vertically-integrated % mass burnt (fire grid)
        firebrand_out : int
            Output flag [0, 1]: firebrand trajectories. Must be 0 when firebrand flag is 0
        emissions_out : int
            Output flag [0, 5]: compressed array emissions (fire grid):
                0 = do not output any emission related variables
                1 = output emissions files and simulate CO in QUIC-SMOKE
                2 = output emissions files and simulate PM2.5 in QUIC- SMOKE
                3 = output emissions files and simulate both CO and PM2.5 in QUIC-SMOKE
                4 = output emissions files but use library approach in QUIC-SMOKE
                5 = output emissions files and simulate both water in QUIC-SMOKE
        radiation_out : int
            Output flag [0, 1]: gridded thermal radiation (fire grid)
        intensity_out : int
            Output flag [0, 1]: surface fire intensity at every fire time step
        """
        InputValidator.positive_integer("nx", nx)
        InputValidator.positive_integer("ny", ny)
        InputValidator.positive_integer("nz", nz)
        InputValidator.negative_one("output_time", output_time)
        if output_time == -1:
            print("CAUTION: User must provide custom times in file QFire_ListOutputTimes.inp when output_time = -1")
        InputValidator.positive_integer("time_now", time_now)
        InputValidator.positive_integer("sim_time", sim_time)
        InputValidator.binary_flag("fire_flag", fire_flag)
        InputValidator.negative_one("random_seed", random_seed)
        InputValidator.positive_integer("fire_time_step", fire_time_step)
        InputValidator.positive_integer("quic_time_step", quic_time_step)
        InputValidator.binary_flag("stretch_grid_flag", stretch_grid_flag)
        InputValidator.string("file_path", file_path)
        if stretch_grid_flag == 1:
            InputValidator.list_of_positive_floats("dz_array", dz_array)
        InputValidator.in_list("fuel_flag", fuel_flag, [1,2,3,4])
        if fuel_flag == 1:
            InputValidator.list_of_positive_floats("fuel_params", fuel_params)
        InputValidator.in_list("ignition_flag", ignition_flag, [1,2,3,4,5,7])
        if ignition_flag in [1,2,3,4,5]:
            InputValidator.list_of_positive_ints("ignition_params", ignition_params)
        InputValidator.positive_integer("ignitions_per_cell", ignitions_per_cell)
        InputValidator.binary_flag("firebrand_flag", firebrand_flag)
        InputValidator.binary_flag("auto_kill", auto_kill)
        # Output flags
        InputValidator.binary_flag("eng_to_atm_out", eng_to_atm_out)
        InputValidator.binary_flag("react_rate_out", react_rate_out)
        InputValidator.binary_flag("fuel_dens_out", fuel_dens_out)
        InputValidator.binary_flag("QF_wind_out", QF_wind_out)
        InputValidator.binary_flag("QU_wind_inst_out", QU_wind_inst_out)
        InputValidator.binary_flag("QU_wind_avg_out", QU_wind_avg_out)
        InputValidator.binary_flag("fuel_moist_out", fuel_moist_out)
        InputValidator.binary_flag("mass_burnt_out", mass_burnt_out)
        InputValidator.binary_flag("firebrand_out", firebrand_out)
        if firebrand_out == 1 and firebrand_out == 0:
            raise ValueError("Firebrand trajectories cannot be output when firebrands are off")
        InputValidator.in_list("emissions_out", emissions_out, [0,1,2,3,4,5])
        InputValidator.binary_flag("radiation_out", radiation_out)
        InputValidator.binary_flag("intensity_out", intensity_out)

        super().__init__("QUIC_fire.inp")
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.output_time = output_time
        self.time_now = time_now
        self.sim_time = sim_time
        self.fire_flag = fire_flag
        self.random_seed = random_seed
        self.fire_time_step = fire_time_step
        self.quic_time_step = quic_time_step
        self.stretch_grid_flag = stretch_grid_flag
        self.stretch_grid_input = self._get_custom_stretch_grid()
        self.file_path = file_path
        self.dz_array = dz_array if dz_array else []
        self.fuel_flag = fuel_flag
        self.fuel_params = fuel_params if fuel_params else []
        self.fuel_density, self.fuel_moisture, self.fuel_height = self._get_fuel_inputs() if fuel_flag ==1 else (None,None,None)
        self.ignition_flag = ignition_flag
        self.ignition_params = ignition_params if ignition_params else []
        self.ignition_locations = self._get_ignition_locations()
        self.ignitions_per_cell = ignitions_per_cell
        self.firebrand_flag = firebrand_flag
        self.auto_kill = auto_kill
        # Output flags
        self.eng_to_atm_out = eng_to_atm_out
        self.react_rate_out = react_rate_out
        self.fuel_dens_out = react_rate_out
        self.QF_wind_out = QF_wind_out
        self.QU_wind_inst_out = QU_wind_inst_out
        self.QU_wind_avg_out = QU_wind_avg_out
        self.fuel_moist_out = fuel_moist_out
        self.mass_burnt_out = mass_burnt_out
        self.emissions_out = emissions_out
        self.radiation_out = radiation_out
        self.intensity_out = intensity_out


    def _get_fuel_inputs(self):
        """
        Writes custom fuel inputs to QUIC_fire.inp, if provided.
        """
        if len(self.fuel_params) != 3:
                raise ValueError("fuel_params must have length of 3")
        # Uniform fuel properties
        if self.fuel_flag == 1:
            fuel_density = f"\n{str(self.fuel_params[0])}"
            fuel_moisture = f"\n{str(self.fuel_params[1])}"
            fuel_height = (f"\n{self.fuel_flag}\t! fuel height flag: 1 = uniform; "
                           f"2 = provided thru QF_FuelMoisture.inp, 3 = Firetech"
                           f" files for quic grid, 4 = Firetech files for "
                           f"different grid (need interpolation)"
                           f"\n{str(self.fuel_params[2])}")
        # Custom fuel .dat files (fuel flags 3 or 4)
        else:
            fuel_density, fuel_moisture, fuel_height = "", "", ""
        
        if self.fuel_flag == 2:
            print("CAUTION: User must provide fuel inputs in QF_FuelDensity.inp, QF_FuelMoisture.inp, and QF_FuelHeight.inp when fuel_flag = 2")

        return fuel_density, fuel_moisture, fuel_height
    
    def _get_ignition_locations(self):
        if self.ignition_flag == 1:
            self._get_ignitions_rect()
        elif self.ignition_flag == 2:
            self._get_ignitions_sq_ring()
        elif self.ignition_flag == 3:
            self._get_ignitions_cir_ring()

        if self.ignition_flag == 4:
            print("CAUTION: User must provide ignition locations in QF_Ignitions.inp when ignition_flag = 4")
        if self.ignition_flag == 5:
            print("CAUTION: User must provide time- and space-dependent ignition locations in QF_IgnitionPattern.inp when ignition_flag = 5")
        
        return ""
    
    def _get_ignitions_rect(self):
        if len(self.ignition_params) != 4:
            raise ValueError("ignition_params must have length of 4 when ignition_flag = 1 (rectangle ignition)")
        x_sw = self.ignition_params[0]
        y_sw = self.ignition_params[1]
        x_len = self.ignition_params[2]
        y_len = self.ignition_params[3]

        if x_sw+x_len > self.nx*2 or y_sw+y_len > self.ny*2:
            raise ValueError("Ignitions outside burn domain")
        
        return (f"\n{str(x_sw)}\t! South-west corner in the x-direction (m)"
                f"\n{str(y_sw)}\t! South-west corner in the y-direction (m)"
                f"\n{str(x_len)}\t! Length in the x-direction (m)"
                f"\n{str(y_len)}\t! Length in the y-direction (m)")
        
    def _get_ignitions_sq_ring(self):
        if len(self.ignition_params) != 6:
            raise ValueError("ignition_params must have length of 6 when ignition_flag = 2 (square ring ignition)")
        x_sw = self.ignition_params[0]
        y_sw = self.ignition_params[1]
        x_len = self.ignition_params[2]
        y_len = self.ignition_params[3]
        x_wid = self.ignition_params[4]
        y_wid = self.ignition_params[5]
    
        if x_sw+x_len > self.nx*2 or y_sw+y_len > self.ny*2:
                raise ValueError("Ignitions outside burn domain")
        
        return (f"\n{str(x_sw)}\t! South-west corner in the x-direction (m)"
                f"\n{str(y_sw)}\t! South-west corner in the y-direction (m)"
                f"\n{str(x_len)}\t! Length in the x-direction (m)"
                f"\n{str(y_len)}\t! Length in the y-direction (m)"
                f"\n{str(x_wid)}\t! Width of the ring in the x-direction (m)"
                f"\n{str(y_wid)}\t! Width of the ring in the y-direction (m)")
    
    def _get_ignitions_cir_ring(self):
        if len(self.ignition_params) != 5:
            raise ValueError("ignition_params must have length of 5 when ignition_flag = 3 (circular ring ignition)")
        x_sw = self.ignition_params[0]
        y_sw = self.ignition_params[1]
        x_len = self.ignition_params[2]
        y_len = self.ignition_params[3]
        wid = self.ignition_params[4]
    
        if x_sw+x_len > self.nx*2 or y_sw+y_len > self.ny*2:
                raise ValueError("Ignitions outside burn domain")
        
        return (f"\n{str(x_sw)}\t! South-west corner in the x-direction (m)"
                f"\n{str(y_sw)}\t! South-west corner in the y-direction (m)"
                f"\n{str(x_len)}\t! Length in the x-direction (m)"
                f"\n{str(y_len)}\t! Length in the y-direction (m)"
                f"\n{str(wid)}\t! Width of the ring (m)")


    def _get_custom_stretch_grid(self):
        """
        Writes a custom stretch grid to QUIC_fire.inp, if provided.
        """
        if self.stretch_grid_flag == 1:
            # Verify that dz_array is not empty
            if not self.dz_array:
                raise ValueError("dz_array must not be empty if stretch_grid_flag "
                                "is 1. Please provide a dz_array with nz elements"
                                " or use a different stretch_grid_flag.")

            # Verify that nz is equal to the length of dz_array
            if self.nz != len(self.dz_array):
                raise ValueError(f"nz must be equal to the length of dz_array. "
                                f"{self.nz} != {len(self.dz_array)}")

            # Write dz_array lines
            dz_array_lines_list = []
            for dz in self.dz_array:
                dz_array_lines_list.append(f"{float(dz)}")
            dz_array_lines = "\n".join(dz_array_lines_list)

            return f"{dz_array_lines}"
        else:
            return self.nz
class Param:

    def __init__(self):
        self.imax = 5000
        self.Lambda = 1.0e+0
        self.LambdaMin = 1.0e-6
        self.bmax = 20
        self.eps = 1.0e-5
        self.gammaInit = 1.0e-2
        self.gammaMin = 1.0e-7
        self.isOracleRandomized = True
        self.tau = 0.9
        self.etaInit = 1.0
        self.etau = 0.9
        self.isDebug = False
        self.mode = "normal"
        # self.mode = "develop"
        # self.methods = ["ipm_exact", "ipm_inexact", "cda_exact", "cda_inexact", "cda_special"]
        self.methods = ["ipm_exact"]
        # self.methods = ["cda_special"]

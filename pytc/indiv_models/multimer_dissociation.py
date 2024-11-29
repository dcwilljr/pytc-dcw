"""
multimer_dissociation.py

A simple dissociation model for an multimer with "multi" copies (e.g. tetramer where multi=4).
Uses numpy.roots to find the roots of the polynomial, assuming that the last root is
the biologically relevant one. Appears to work for dimers and tetramers.
"""
__author__ = "David C. Williams Jr."
__date__ = "2024-11-28"


import numpy as np

from pytc.indiv_models.base import ITCModel

class MultimerDissociationModel(ITCModel):
    """
    Multimer dissociation model. 
    """
    def param_definition(K=5.48e-6,dH=31800.0, offset = 1400):
        pass
    
    def __init__(self,
                 multi=2,
                 S_cell=0.0,S_syringe=0.0,
                 T_cell=0.0,   T_syringe=1000e-6,
                 cell_volume=300.0,
                 shot_volumes=[2.5 for i in range(30)]):

        """
        multi: number of molecules in multimer
        S_cell: stationary concentration in cell in M
        S_syringe: stationary concentration in syringe in M
        T_cell: titrant concentration cell in M
        T_syringe: titrant concentration syringe in M
        cell_volume: cell volume, in uL
        shot_volumes: list of shot volumes, in uL.
        """

        self._multi = multi
        super().__init__(S_cell,S_syringe,T_cell,T_syringe,cell_volume,shot_volumes)


    @property
    def dQ(self):
        """
        Calculate the heats (in microcalories) across shots for a 
        given dissociation enthalpy ("dH" - cal) and 
        dissociation constant ("K" = 1/Kassociation). 
        
        The resulting K is in units of M (monomeric disocciation constant), 
        or equivalent to the individual K in a stepwise dissociation, 
        with all K being equal. 
                K              K              K
            PN <-> P(N-1) + P <-> P(N-2) + P <-> ... P 

                     K^(N-1)
                   PN <---> NP       K^(N-1) = [P]^N/[PN] 

                        1/K^(N-1) = Kass = [PN]/[P]^N
                        Kass * [P]^N = [PN] = (Ptot - [P])/N
                        N*Kass*[P]^N + [P] - Ptot = 0
        
        The "offset" is a linear baseline offset of heat in cal per mol 
        of injected monomer.            
        """
        
        #----------Set the multimerization state, m >= 2 -----------------#
        m = self._multi
        
        #----------Find roots for polynomial describing free protein concentration (Pfree)----------------#
        def find_root(Ptot, Kmonomer, m):
            if m == 2:
                return ((np.sqrt(1 + (8 * Kmonomer * Ptot)) - 1)/(4 * Kmonomer))
            else:
                K = np.power(Kmonomer, m-1)
                coeff = [0 for i in range(m + 1)]
                coeff[m] = -1*Ptot
                coeff[m-1] =  1.0
                coeff[0] = (m * K)
                roots = np.roots(coeff)
                return np.real(roots[len(roots)-1])  
                     # np.roots() appears to return ordered array so that we need the last
        
        #----------Get paramater values, syringe concentration, shot volumes, and cell volume -----------------#
    
        dH = self.param_values["dH"]
        Kmonomer = 1/self.param_values["K"]
        offset = self.param_values["offset"]
        syringe_conc = self._T_syringe
        
        cell_volume = self._cell_volume*1e-6
        shot_vol = self._shot_volumes*1e-6
        
        #-----Calculate total, multimeric, and free protein concentrations after each shot, before and after dissociation--------#
        
        Pfree_syr = find_root(syringe_conc, Kmonomer, m)

        total_injected = 0
        lambda_i = [0]

        for i in range(len(shot_vol)):
            total_injected = total_injected + shot_vol[i]
            Pfree_pre = (Pfree_syr * (total_injected/cell_volume) * (1 - (total_injected/(2*cell_volume))))
            Ptot = (syringe_conc * (total_injected/cell_volume) * (1 - (total_injected/(2*cell_volume))))
            Pfree = find_root(Ptot, Kmonomer, m)   
            lambda_i.append(Pfree - Pfree_pre)        

        #-----Calculate heat generated per each shot--------#
        
        delta_heat = []
        
        for j in range(1,len(lambda_i)):
            heatj = (dH * cell_volume *((lambda_i[j] - lambda_i[j-1]) + ((shot_vol[j-1]/(2 * cell_volume)) * (lambda_i[j] + lambda_i[j-1]))))
            delta_heat.append(heatj * 1e6)
    
        #-----Calculate an offset due to heat associated with each injection (per mol injected)--------#
        
        offsets = shot_vol * syringe_conc * offset * 1e6

        to_return = np.array(delta_heat) - offsets

        return to_return
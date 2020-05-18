#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:04:10 2020

@author: ramyagurunathan

Implementation of the model in O. Weis 1979

Return Transmission Coefficient

Will likely need ot utilize christoffel package

e3 is going ot be ther interface normal for now

Currently, assums side 1 and side 2 have the same density
"""

import sys
sys.path.append('/Users/ramyagurunathan/Documents/PhDProjects/BoundaryScattering/scattering_scripts')

from christoffel import Christoffel
import christoffel as ch
from AMMTransport import AMMTransport
import numpy as np
from math import sin, cos, pi, nan

'''
Bi2Te3 stiffness matrix
'''

stiffness = np.array([[74.4, 21.7, 27.0, 13.3, 0.0, 0.0], [21.7, 74.4, 27.0, -13.3, 0.0, 0.0],\
        [27.0, 27.0, 47.7, 0.0, 0.0, 0.0], [13.3, -13.3, 0.0, 27.4, 0.0, 0.0],\
        [0.0, 0.0, 0.0, 0.0, 27.4, 13.3], [0.0, 0.0, 0.0, 0.0, 13.3, 26.4]])
density = 7480

'''
Class Initialization
'''

def v_mag(v_vector):
    return (v_vector[0]**2 + v_vector[1]**2 + v_vector[2]**2)**(1/2)

def delta(x,y):
    if x==y:
        r = 1
    if x!=y:
        r = 0
    return r

class InterfaceTransmission:
    def __init__(self, stiffness, density, interface_norm = [0, 0, 1]):
        self.ch_obj = ch.Christoffel(stiffness, density)
        self.interface_norm = np.array(interface_norm)
        #Unlike Christoffel, don't need to divide by desnity. Can see this by comparing
        #eigenvalue equations
        self.rank4c = np.array(ch.de_voigt(stiffness)) * 1000
        #Set the z_direction to (0,0,1) and the x_direction to (1,0,0)
        self.ch_obj.rotate_tensor(x_dir = [1.0, 0.0, 0.0], z_dir = [0.0, 0.0, 1.0])
        self.density = density

    '''
    Calculate the Refelection and transmission factors from O. Weis Model
    '''
    '''
    Calculate the allowed plane waves from an incident wave 
    '''
    def e_ll(self, phi):
        '''
        Essentially set theta to 0
        '''
        return np.array([-cos(phi), -sin(phi), 0])
    
    def e_n_sigmaA(self, theta, phi):
        return np.array([-cos(phi) * sin(theta), -sin(phi)*sin(theta), -cos(theta)])
        
    def g(self, pol, theta, phi):
        '''
        Reciprocal trace velocity along the boundary
        I think this should be a scalar quantitiy
        pol: 2 is primary, 1 is fast secondary, 0 is slow secondary
        (basically, velocities sorted from low to high)
        '''
        e_n_sigmaA = self.e_n_sigmaA(theta, phi)
        self.ch_obj.set_direction_cartesian(e_n_sigmaA)
        c_sigmaA = self.ch_obj.get_phase_velocity()
        return sin(theta) / (c_sigmaA[pol])
                
    def symm_tensor_B(self):
        '''
        I thought this might be the inner product
        But it seems like it is two dot products?
        '''
        dot1 = np.dot(self.interface_norm, self.rank4c)
        dot2 = np.dot(self.interface_norm, dot1)
        return dot2
    
    def symm_tensor_E(self, pol, theta, phi):
        dot1 = np.dot(np.dot(self.interface_norm, self.rank4c), self.e_ll(phi))
        dot2 = np.dot(np.dot(self.e_ll(phi), self.rank4c), self.interface_norm)
        return np.dot(self.g(pol, theta, phi), dot1 + dot2)
    
    def symm_tensor_F(self, pol, theta, phi):
        dot1 = np.dot(self.g(pol, theta, phi)**2, self.e_ll(phi))
        dot2 = np.dot(np.dot(dot1, self.rank4c), self.e_ll(phi))
        return dot2 - self.density * np.identity(3)
    
    def solve_h_tau(self, pol, theta, phi):
        def calculate_coeffs(self, tau, phi):
            F = self.symm_tensor_F(pol,theta, phi)
            E = self.symm_tensor_E(pol,theta, phi)
            B = self.symm_tensor_B()
            X0 = F[1,1] * F[2,2] - F[1,2]**2
            X1 = E[1,1] * F[2,2] + F[1,1]* E[2,2] - 2 * E[1,2] * F[1,2]
            X2 = B[1,1] * F[2,2] - 2 * B[1,2] * F[1,2] + E[1,1] * E[2,2] -\
            E[1,2]**2 + F[1,1] * B[2,2]
            X3 = B[1,1] * E[2,2] + B[2,2] * E[1,1] - 2 * B[1,2] * E[1,2]
            X4 = B[1,1] * B[2,2] - B[1,2]**2
            Y0 = F[0,1] * F[2,2] - F[0,2] * F[1,2]
            Y1 = E[0,1] * F[2,2] + F[0,1] * E[2,2] - E[0,2] * F[1,2] - E[1,2] * F[0,2]
            Y2 = B[0,1] * F[2,2] + E[0,1] * E[2,2] + B[2,2] * F[0,1] - B[1,2] * F[0,2] -\
            B[0,2] * F[1,2] - E[0,2] * E[1,2]
            Y3 = B[0,1] * E[2,2] + E[0,1] * B[2,2] - E[0,2] * B[1,2] - B[0,2] * E[1,2]
            Y4 = B[0,1] * B[2,2] - B[0,2] * B[1,2]
            Z0 = F[0,1] * F[1,2] - F[0,2] * F[1,1]
            Z1 = E[0,1] * F[1,2] + E[1,2] * F[0,1] - E[0,2] * F[1,1] - E[1,1] * F[0,2]
            Z2 = B[0,1] * F[1,2] + B[1,2] * F[0,1] + E[0,1] * E[1,2] - \
            B[1,1] * F[0,2] - B[0,2] * F[1,1] - E[0,2] * E[1,1]
            Z3 = B[0,1] * E[1,2] + B[1,2] * E[0,1] - B[1,1] * E[0,2] - B[0,2] * E[1,1]
            Z4 = B[0,1] * B[1,2] - B[0,2] * B[1,1]
            #Coefficients
            c0 = F[0,0] * X0 - F[0,1] * Y0 + F[0,2] * Z0
            c1 = F[0,0] * X1 + E[0,0] * X0 - F[0,1] * Y1 - E[0,1] * Y0 + F[0,2] * Z1 +\
            E[0,2] * Z0
            c2 = B[0,0] * X0 + F[0,0] * X2 + E[0,0] * X1 - B[0,1] * Y0 - F[0,1] * Y2 -\
            E[0,1] * Y1 + B[0,2] * Z0 + F[0,2] * Z2 + E[0,2] * Z1
            c3 = B[0,0] * X1 + F[0,0] * X3 + E[0,0] * X2 - B[0,1] * Y1 - F[0,1] * Y3 -\
            E[0,1] * Y2 + B[0,2] * Z1 + F[0,2] * Z3 + E[0,2] * Z2
            c4 = B[0,0] * X2 + F[0,0] * X4 + E[0,0] * X3 - B[0,1] * Y2 - F[0,1] * Y4 -\
            E[0,1] * Y3 + B[0, 2] * Z2 + F[0,2] * Z4 + E[0,2] * Z3
            c5 = B[0,0] * X3 + E[0,0] * X4 - B[0,1] * Y3 - E[0,1] * Y4 + B[0,2] * Z3 +\
            E[0,2] * Z4
            c6 = B[0,0] * X4 - B[0,1] * Y4 + B[0,2] * Z4
            return [c6, c5, c4, c3, c2, c1, c0]
        coeff = calculate_coeffs(self, theta, phi)
        h_tau = np.roots(coeff)
        return h_tau
            
    def select_solutions(self, h_tau, pol, theta, phi):
        '''
        Think I just need to pick out plane wave solutions. Only these should have
        non-zero transmittance
        '''
        plane_wave_soln = {'h' : [], 'e_n_tauB' : [], 'v_p' : []}
        for real, h in zip(np.isreal(h_tau), h_tau):
            if real:
                plane_wave_soln['h'].append(h)
                e_n_tauB = self.g(pol, theta, phi) * self.e_ll(phi) + h * self.interface_norm
                c_tau = 1 / np.sqrt(self.g(pol, theta, phi)**2 + h**2)
                plane_wave_soln['e_n_tauB'].append(e_n_tauB)
                plane_wave_soln['v_p'].append(c_tau)
        def match_polarization(plane_wave_soln):
            pol_list = []
            for pw in range(len(plane_wave_soln['h'])):
                self.ch_obj.set_direction_cartesian(plane_wave_soln['e_n_tauB'][pw])
                v_p = self.ch_obj.get_phase_velocity()
                abs_diff = lambda list_value : abs(list_value - plane_wave_soln['v_p'][pw])
                match = min(v_p, key=abs_diff)
                pol_list.append(list(v_p).index(match))
            return pol_list
        plane_wave_soln['pol'] = match_polarization(plane_wave_soln)
        #check condition 3.5 in Weis paper
        def check_conditions(pw_soln): # should maybe include all solns that satisfy condition...?
            cond_list = []
            vg_tauB_list = []
            for pw in range(len(pw_soln['h'])): 
                self.ch_obj.set_direction_cartesian(pw_soln['e_n_tauB'][pw])
                vg_list = self.ch_obj.get_group_velocity()
                vg_tauB = vg_list[pw_soln['pol'][pw]]
                vg_tauB_list.append(vg_tauB)
                if np.dot(self.interface_norm, vg_tauB) <= 0:
                    cond_list.append(pw)
            return cond_list, vg_tauB_list
        plane_wave_soln['condition'], plane_wave_soln['v_g_tauB'] =\
        check_conditions(plane_wave_soln)
        return plane_wave_soln
    
    
    def single_phonon_trans_reflect(self, inc_pol, theta, phi, reflection = True):
        '''
        Calculate the transmisssivity for a single incident phonon
        
        Return transmissivity averaged over different excited polarizations
        
        pol_specific: will give transmission functions for each polarization of the
        excited wave
        
        default: returns the average transmissivity?
        
        Not sure how I should be handling cases where the group velocity has a
        negative z value...
        
        Need to check what is meant by the negative transmissivity and reflectivity would
        really mean
        '''
        #First, without the amplitude weighting factor (need to figure that out)
        h_tau = self.solve_h_tau(inc_pol, theta, phi)
        #Double-check that this is the correct incident velocity   
        e_n_sigmaA = self.e_n_sigmaA(theta, phi)
#        print(e_n_sigmaA)
        self.ch_obj.set_direction_cartesian(e_n_sigmaA)
        v_g = self.ch_obj.get_group_velocity()
        v_g_sigmaA = v_g[inc_pol]
        #Get the polarization vector of the incident phonon
        eigen_vec = self.ch_obj.get_eigenvec()
        e_sigma = eigen_vec[inc_pol]
        pw_soln = self.select_solutions(h_tau, inc_pol, theta, phi)
        trans = 0
        trans_pol = {}
        reflect_pol = {}
        for indx in range(len(pw_soln['h'])):
            #Need to add density ratio if side 1 and side 2 have differnet rho
            if indx in pw_soln['condition']:
                t = np.dot(self.interface_norm, pw_soln['v_g_tauB'][indx]) * self.density /\
                (np.dot(self.interface_norm, v_g_sigmaA) * self.density)
                if t < 0:
                    trans_pol[str(pw_soln['pol'][indx])] = nan
                else:
                    trans_pol[str(pw_soln['pol'][indx])] = t
            else:
                r = -1 * np.dot(self.interface_norm, pw_soln['v_g_tauB'][indx]) /\
                np.dot(self.interface_norm, v_g_sigmaA)
                if r < 0:
                    reflect_pol[str(pw_soln['pol'][indx])] = nan
                else:
                    reflect_pol[str(pw_soln['pol'][indx])] = r
        for pol in ['0', '1', '2']:
            if pol not in reflect_pol.keys():
                reflect_pol[pol] = 0
            if pol not in trans_pol.keys():
                trans_pol[pol] = 0
#            print(v_g_sigmaA)
#            print(t)
#            if t < 0:
#                '''
#                Negative transmissivities should mean that group velocity was not
#                in direction of the interface
#                '''
#                trans_pol[str(pw_soln['pol'][indx])] = nan
#                continue
#            trans = trans + t
#            trans_pol[str(pw_soln['pol'][indx])] = t
#        if pol_specific:
#            transmissivity = trans_pol
#            reflectivity = reflect_pol
#        else:
#            transmissivity = trans / len(pw_soln['condition'])
#            reflectivity = 
        return trans_pol, reflect_pol
    
    #Need to adjust to take average over the polarization   
    def transmission_factor(self, n_angle):
        '''
        Return: 
            matrix of tramission factors for each polarization 
            by default, transmission from side B to side A?
        Input:
            theta and phi values     
            
        some values are very large
        '''
        d_angle = (pi / 2) / n_angle    
        theta_list = np.arange(d_angle, pi / 2 + d_angle, d_angle)
        phi_list = np.arange(d_angle, 2 * pi, d_angle)
        running_integrand = 0
        for theta in theta_list:
            for phi in phi_list:
                for pol in [0,1,2]:
                    trans = self.single_phonon_transmissivity(pol, theta - (d_angle / 2),\
                                                         phi - d_angle / 2)
                    print(trans)
                    running_integrand = running_integrand + trans * np.sin(theta - (d_angle / 2)) *\
                    d_angle**2
        return running_integrand
    
    def amplitude_factors(self, inc_pol, theta, phi, pw_soln):
        '''
        set up system of equations to get the amplitude factors
        
        L:longitudinal
        T1: fast transverse
        T2: slow transverse
        '''
        e_norm = self.e_n_sigmaA(theta, phi)
        e_ll = self.e_ll(phi)
        g = self.g(inc_pol, theta, phi)
        eLA = e_norm
        eLB = None
        eT1B = None
        eT2B = None
        for c in pw_soln['condition']: #this will only cpature the transverse waves
            pol = pw_soln['pol'][c]
            if pol == 0:
                eLB = pw_soln['e_n_tauB'][c]
                vp_LB = pw_soln['v_p'][c]
                h_L = pw_soln['h'][c]
            if pol == 1:
                eT1B = pw_soln['e_n_tauB'][c]
                vp_T1B = pw_soln['v_p'][c]
                h_T1 = pw_soln['h'][c]
            if pol == 2:
                eT2B = pw_soln['e_n_tauB'][c]
                vp_T2B = pw_soln['v_p'][c]
                h_T2 = pw_soln['h'][c]
        if eLB is None: eLB = 0
        if eT1B is None: eT1B = 0
        if eT2B is None: eT2B = 0  
        refl_waves = [n for n in range(len(pw_soln['h'])) if n not in pw_soln['condition']]
        for c in refl_waves: #this will only cpature the reflected waves
            pol = pw_soln['pol'][c]
            if pol == 0:
                eLA2 = pw_soln['e_n_tauB'][c]
                vp_LA2 = pw_soln['v_p'][c]
            if pol == 1:
                eT1A2 = pw_soln['e_n_tauB'][c]
                vp_T1A2 = pw_soln['v_p'][c]
            if pol == 2:
                eT2A2 = pw_soln['e_n_tauB'][c]
                vp_T2A2 = pw_soln['v_p'][c]
        if eLB is None: eLB = 0
        if eT1B is None: eT1B = 0
        if eT2B is None: eT2B = 0            
        eqn_matrix = np.zeros([3,6])
        t, r = self.single_phonon_trans_reflect(inc_pol, theta, phi)
        for x in [0,1,2]:
            eqn_matrix[x,:] = [eLA2[x], eT1A2[x], eT2A2[x], eLB[x], eT1B[x], eT2B[x]]
        print(eqn_matrix)
        #This is from Eqn 4.3 but realizing it's just one equation..
#        eqn_matrix[3, :] = r['2'] + r['1'] + r['0'] + t['2'] + t['1'] + t['0']
        eqn_matrix_2 = np.zeros([3,6,3,3])
        for y in [0,1,2]:
            '''
            Longitudinal
            '''
            tauL_m1 = self.rank4c[2,y,:,0] * eLA2[0] * eLA2[0]/\
            vp_LA2
            tauL_m2 = self.rank4c[2,y,:,1] * eLA2[0] * eLA2[1]/\
            vp_LA2
            tauL_m3 = self.rank4c[2,y,:,2] * eLA2[0] * eLA2[2]/\
            vp_LA2
            term1_L = np.array([tauL_m1, tauL_m2, tauL_m3])
            tauL2_m1 = self.rank4c[2,y,:,0] * eLB[0] * (g * e_ll[0] + h_T1 * delta(2,0))
            tauL2_m2 = self.rank4c[2,y,:,1] * eLB[0] * (g * e_ll[1] + h_T1 * delta(2,1))
            tauL2_m3 = self.rank4c[2,y,:,2] * eLB[0] * (g * e_ll[2] + h_T1 * delta(2,2))
            term2_L = np.array([tauL2_m1, tauL2_m2, tauL2_m3])
            '''
            Fast Transverse
            '''
            tauT1_m1 = self.rank4c[2,y,:,0] * eT1A2[0] * eT1A2[0]/\
            vp_T1A2
            tauT1_m2 = self.rank4c[2,y,:,1] * eT1A2[0] * eT1A2[1]/\
            vp_T1A2
            tauT1_m3 = self.rank4c[2,y,:,2] * eT1A2[0] * eT1A2[2]/\
            vp_T1A2
            term1_T1 = np.array([tauT1_m1, tauT1_m2, tauT1_m3])
            tauT1_2_m1 = self.rank4c[2,y,:,0] * eT1B[0] * (g * e_ll[0] + h_T2 * delta(2,0))
            tauT1_2_m2 = self.rank4c[2,y,:,1] * eT1B[0] * (g * e_ll[1] + h_T2 * delta(2,1))
            tauT1_2_m3 = self.rank4c[2,y,:,2] * eT1B[0] * (g * e_ll[2] + h_T2 * delta(2,2))
            term2_T1 = np.array([tauT1_2_m1, tauT1_2_m2, tauT1_2_m3])
            '''
            Slow Transverse
            '''
            tauT2_m1 = self.rank4c[2,y,:,0] * eT2A2[0] * eT2A2[0]/\
            vp_T2A2
            tauT2_m2 = self.rank4c[2,y,:,1] * eT2A2[0] * eT2A2[1]/\
            vp_T2A2
            tauT2_m3 = self.rank4c[2,y,:,2] * eT2A2[0] * eT2A2[2]/\
            vp_T2A2
            term1_T2 = np.array([tauT2_m1, tauT2_m2, tauT2_m3])
            tauT2_2_m1 = self.rank4c[2,y,:,0] * eT2B[0] * (g * e_ll[0] + h_L * delta(2,0))
            tauT2_2_m2 = self.rank4c[2,y,:,1] * eT2B[0] * (g * e_ll[1] + h_L * delta(2,1))
            tauT2_2_m3 = self.rank4c[2,y,:,2] * eT2B[0] * (g * e_ll[2] + h_L * delta(2,2))
            term2_T2 = np.array([tauT2_2_m1, tauT2_2_m2, tauT2_2_m3])            
            '''
            Equation Coefficient Tensor
            '''
            eqn_matrix_2[y,:] = [term1_L, term1_T1, term1_T2, term2_L, term2_T1, term2_T2]            
        print(eqn_matrix_2)                
        vars_vector = [RL, RT1, RT2, TL, TT1, TT2]
            '''
            Solution Tensor
            '''

if __name__ == '__main__':
    trans = InterfaceTransmission(stiffness, density)
    theta = pi/7
    phi = pi/7
    F = trans.symm_tensor_F(1,theta, phi)
    E = trans.symm_tensor_E(1,theta, phi)
    B = trans.symm_tensor_B()
    h_tau = trans.solve_h_tau(1,theta, phi)
    pw_soln = trans.select_solutions(h_tau, 1, theta, phi)
    t_single, r_single = trans.single_phonon_trans_reflect(1, theta, phi)
#    alpha = trans.transmission_factor(10)



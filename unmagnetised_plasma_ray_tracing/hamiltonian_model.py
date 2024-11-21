#!/usr/bin/python3

# Standard imports
import logging
import numpy as np

# Local imports

logger = logging.getLogger(__name__)

class UnmagnetisedPlasmaHamiltonian:
    __slots__ = ("X_model",)

    def __init__(self, X_model):
        self.X_model = X_model

    def get_X_N2(self, position, refractive_index):
        X = self.X_model.stix_X(position)
        N2 = np.dot(refractive_index, refractive_index)
        return X, N2
    
    def hamiltonian_function(self, X, N2):
        # return (1 - X) * (1 - X - N2)**2
        return 1 - X - N2

    def hamiltonian_first_derivative_X(self, X, N2):
        # return -(1 - X - N2) * (3 - 3*X - N2)
        return -1

    def hamiltonian_first_derivative_N2(self, X, N2):
        # return -2 * (1 - X) * (1 - X - N2)
        return -1
    
    def hamiltonian_normalised_first_derivative_f_function(self, X, N2):
        # return 2 * (1 - X - N2) * (3 * X * (1 - X - N2) + 2 * N2)
        return 2 * (X + N2)

    def hamiltonian_second_derivative_X(self, X, N2):
        return 0

    def hamiltonian_second_derivative_N2(self, X, N2):
        return 0

    def hamiltonian(self, position, refractive_index):
        X, N2 = self.get_X_N2(position, refractive_index)
        return self.hamiltonian_function(X, N2)

    def hamiltonian_first_derivative_x(self, position, refractive_index):
        X, N2 = self.get_X_N2(position, refractive_index)
        dD_dX = self.hamiltonian_first_derivative_X(X, N2)
        dX_dx = self.X_model.stix_X_first_derivative_x(position)
        return dD_dX * dX_dx

    def hamiltonian_first_derivative_N(self, position, refractive_index):
        X, N2 = self.get_X_N2(position, refractive_index)
        dD_dN2 = self.hamiltonian_first_derivative_N2(X, N2)
        return 2 * dD_dN2 * refractive_index
    
    def hamiltonian_normalised_first_derivative_f(self, position, refractive_index):
        X, N2 = self.get_X_N2(position, refractive_index)
        return 2 * (X + N2)

    def hamiltonian_second_derivative_x(self, position, refractive_index):
        d2X_dx2 = self.X_model.stix_X_second_derivative_x(position)
        return -d2X_dx2

    def hamiltonian_second_derivative_N(self, position, refractive_index):
        d2X_dx2 = self.X_model.stix_X_second_derivative_x(position)
        return -2 * np.identity(3)

    def hamiltonian_second_derivative_xN(self, position, refractive_index):
        n = len(position)
        return np.zeros((n, n))
    
    def hamiltonian_normalised_second_derivative_xf(self, position, refractive_index):
        dX_dx = self.X_model.stix_X_first_derivative_x(position)
        return 2 * dX_dx

    def hamiltonian_normalised_second_derivative_Nf(self, position, refractive_index):
        return 4 * refractive_index

    def normalised_ray_velocity(self, position, refractive_index):
        # X, N2 = self.get_X_N2(position, refractive_index)
        # dX_dx = self.X_model.stix_X_first_derivative_x(position)
        dD_dx = self.hamiltonian_first_derivative_x(position, refractive_index)
        dD_dN = self.hamiltonian_first_derivative_N(position, refractive_index)
        f_dD_df = self.hamiltonian_normalised_first_derivative_f(position, refractive_index)

        n = len(position)
        v_norm = np.zeros(2 * n)
        v_norm[:n] = -dD_dN / f_dD_df
        v_norm[n:] = dD_dx / f_dD_df

        return v_norm

    def normalised_ray_acceleration(self, position, refractive_index):
        # X, N2 = self.get_X_N2(position, refractive_index)
        # dX_dx = self.X_model.stix_X_first_derivative_x(position)
        # d2X_dx2 = self.X_model.stix_X_second_derivative_x(position)

        dD_dx = self.hamiltonian_first_derivative_x(position, refractive_index)
        dD_dN = self.hamiltonian_first_derivative_N(position, refractive_index)
        f_dD_df = self.hamiltonian_normalised_first_derivative_f(position, refractive_index)

        d2D_dx2 = self.hamiltonian_second_derivative_x(position, refractive_index)
        d2D_dN2 = self.hamiltonian_second_derivative_N(position, refractive_index)
        # f_d2D_dxdf = self.hamiltonian_normalised_second_derivative_xf(position, refractive_index)
        # f_d2D_dNdf = self.hamiltonian_normalised_second_derivative_Nf(position, refractive_index)

        n = len(position)
        ray_acceleration = np.zeros(2 * n)
        ray_acceleration[:n] = -np.einsum('ij,j', d2D_dN2, dD_dx) / f_dD_df**2
        ray_acceleration[n:] = -np.einsum('ij,j', d2D_dx2, dD_dN) / f_dD_df**2
        # ray_acceleration[:3] = -2 * dX_dx / f_dD_df**2
        # ray_acceleration[3:] = -2 * np.dot(d2X_dx2, refractive_index) / f_dD_df**2

        return ray_acceleration

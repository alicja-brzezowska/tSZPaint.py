import astropy.constants as aconst
import jax
import jax.numpy as jnp
import unxt as u

jax.config.update("jax_enable_x64", True)

G_CGS_ASTRO = aconst.G.cgs

G_CGS = u.Quantity(jnp.asarray(G_CGS_ASTRO.value), G_CGS_ASTRO.unit).value
G_SI = u.Quantity(jnp.asarray(aconst.G.value), aconst.G.unit)

# prefactor for compton_y : Thomson cross-section / (electron mass * c^2)
P_E_FACTOR: float = aconst.sigma_T.value / (aconst.m_e.value * aconst.c.value**2)

M_SUN = u.Quantity(aconst.M_sun.value, aconst.M_sun.unit).value

G_CM3_TO_MSUN_MPC3 = u.Quantity(1.0, "g/cm3").to("Msun/Mpc3").value
G_CM3_TO_KG_M3 = u.Quantity(1.0, "g/cm3").to("kg/m3").value

"""
M500c → M200m mass conversion for AbacusSummit (Planck 2018) cosmology,
using colossus (Diemer 2018, ApJS 239, 35).

The input threshold MAX_M500C_MSUN is in physical M_sun (as quoted in
Hilton+2021, ACT DR5 cluster catalog, arxiv:2009.11043).

Limitations relative to Boryana's full requirements (see arxiv:2009.11043 Fig. 7):
  - The 3.8e14 M_sun cut has some redshift dependence that is not captured
    here; a single representative redshift Z_REF is used instead.
"""

from colossus.cosmology import cosmology as colossus_cosmo
from colossus.halo import concentration, mass_defs

colossus_cosmo.setCosmology('planck18')

H_ABACUS = 0.6736  


MAX_M500C_MSUN = 3.8e14   # Hilton+2021 upper M500c cut [M_sun, physical units]
Z_REF          = 0.5    # representative redshift (median of LRG snapshots)

# Concentration model — diemer19 is the default state-of-the-art;
# dutton14 matches the parameterisation in the original hand-coded version.
C_MODEL = 'diemer19'


def m500c_to_m200m(m500c_h: float, z: float, c_model: str = C_MODEL) -> float:
    """Convert M500c [M_sun/h] → M200m [M_sun/h] at redshift z.

    Uses colossus to look up the concentration at M500c then converts
    the mass definition via NFW profile matching.
    """
    c = concentration.concentration(m500c_h, '500c', z, model=c_model)
    m200m_h, _, _ = mass_defs.changeMassDefinition(m500c_h, c, z, '500c', '200m')
    return float(m200m_h)

# Hilton+2021 M500c cut [physical M_sun] → M200m [M_sun/h] at Z_REF.
MAX_M200M_H: float = m500c_to_m200m(MAX_M500C_MSUN * H_ABACUS, Z_REF)


if __name__ == "__main__":
    print(
        f"Input:  M500c = {MAX_M500C_MSUN:.3e} M_sun  [physical, z_ref={Z_REF}]\n"
        f"Output: M200m = {MAX_M200M_H:.6e} M_sun/h\n"
        f"        M200m = {MAX_M200M_H / H_ABACUS:.6e} M_sun  [physical]\n"
        f"c-M model: {C_MODEL}"
    )

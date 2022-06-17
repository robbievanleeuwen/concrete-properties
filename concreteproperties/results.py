from dataclasses import dataclass, field


@dataclass
class ConcreteProperties:
    """Class for storing gross concrete section properties."""

    # section areas
    total_area: float = 0
    concrete_area: float = 0
    steel_area: float = 0
    e_a: float = 0

    # section mass
    mass: float = 0

    # section perimeter
    perimeter: float = 0

    # first moments of area
    e_qx: float = 0
    e_qy: float = 0

    # centroids
    cx: float = 0
    cy: float = 0

    # second moments of area
    e_ixx_g: float = 0
    e_iyy_g: float = 0
    e_ixy_g: float = 0
    e_ixx_c: float = 0
    e_iyy_c: float = 0
    e_ixy_c: float = 0
    e_i11_c: float = 0
    e_i22_c: float = 0

    # principal axis angle
    phi: float = 0

    # section moduli
    e_zxx_plus: float = 0
    e_zxx_minus: float = 0
    e_zyy_plus: float = 0
    e_zyy_minus: float = 0
    e_z11_plus: float = 0
    e_z11_minus: float = 0
    e_z22_plus: float = 0
    e_z22_minus: float = 0

    # plastic properties
    squash_load: float = 0
    tensile_load: float = 0
    axial_pc_x: float = 0
    axial_pc_y: float = 0
    conc_ultimate_strain: float = 0


@dataclass
class TransformedConcreteProperties:
    """Class for storing transformed gross concrete section properties.

    :param concrete_properties: Concrete properties object
    :type concrete_properties:
        :class:`~concreteproperties.concrete_section.ConcreteProperties`
    :param float elastic_modulus: Reference elastic modulus
    """

    concrete_properties: ConcreteProperties = field(repr=False)
    elastic_modulus: float

    # first moments of area
    qx: float = 0
    qy: float = 0

    # second moments of area
    ixx_g: float = 0
    iyy_g: float = 0
    ixy_g: float = 0
    ixx_c: float = 0
    iyy_c: float = 0
    ixy_c: float = 0
    i11_c: float = 0
    i22_c: float = 0

    # section moduli
    zxx_plus: float = 0
    zxx_minus: float = 0
    zyy_plus: float = 0
    zyy_minus: float = 0
    z11_plus: float = 0
    z11_minus: float = 0
    z22_plus: float = 0
    z22_minus: float = 0

    def __post_init__(
        self,
    ):

        self.qx = self.concrete_properties.e_qx / self.elastic_modulus
        self.qy = self.concrete_properties.e_qy / self.elastic_modulus
        self.ixx_g = self.concrete_properties.e_ixx_g / self.elastic_modulus
        self.iyy_g = self.concrete_properties.e_iyy_g / self.elastic_modulus
        self.ixy_g = self.concrete_properties.e_ixy_g / self.elastic_modulus
        self.ixx_c = self.concrete_properties.e_ixx_c / self.elastic_modulus
        self.iyy_c = self.concrete_properties.e_iyy_c / self.elastic_modulus
        self.ixy_c = self.concrete_properties.e_ixy_c / self.elastic_modulus
        self.i11_c = self.concrete_properties.e_i11_c / self.elastic_modulus
        self.i22_c = self.concrete_properties.e_i22_c / self.elastic_modulus
        self.zxx_plus = self.concrete_properties.e_zxx_plus / self.elastic_modulus
        self.zxx_minus = self.concrete_properties.e_zxx_minus / self.elastic_modulus
        self.zyy_plus = self.concrete_properties.e_zyy_plus / self.elastic_modulus
        self.zyy_minus = self.concrete_properties.e_zyy_minus / self.elastic_modulus
        self.z11_plus = self.concrete_properties.e_z11_plus / self.elastic_modulus
        self.z11_minus = self.concrete_properties.e_z11_minus / self.elastic_modulus
        self.z22_plus = self.concrete_properties.e_z22_plus / self.elastic_modulus
        self.z22_minus = self.concrete_properties.e_z22_minus / self.elastic_modulus

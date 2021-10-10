
# selected dynamics parameters that indeed affect the dynamics 
DYNAMICS_PARAMS = {
    'pandapushik2dsimpledynamics': [
    # 'link1_mass', 'link2_mass', 'link3_mass', 'link4_mass', 'link5_mass', 'link6_mass', 'link7_mass', 
    'joint1_damping', 'joint2_damping', 'joint3_damping', 'joint4_damping', 'joint5_damping', 'joint6_damping', 'joint7_damping', 
    'joint1_armature', 'joint2_armature', 'joint3_armature', 'joint4_armature', 'joint5_armature', 'joint6_armature', 'joint7_armature', 
    'actuator_velocity_joint1_kv', 'actuator_velocity_joint2_kv', 'actuator_velocity_joint3_kv', 'actuator_velocity_joint4_kv', 
    'actuator_velocity_joint5_kv', 'actuator_velocity_joint6_kv', 'actuator_velocity_joint7_kv', 
    # 'actuator_position_finger_joint1_kp_1000000', 'actuator_position_finger_joint2_kp_1000000', 
    # 'table_size_0', 'table_size_1', 'table_size_2', 'table_friction_0', 'table_friction_1', 'table_friction_2', 
    'boxobject_size_0', 'boxobject_size_1', 'boxobject_size_2', 
    'boxobject_friction_0', 'boxobject_friction_1', 'boxobject_friction_2', 
    'boxobject_density_1000', 
    # 'pandaik_z_proportional_gain'
    ], 

    'pandapushfkdynamics': [
    # 'link1_mass', 'link2_mass', 'link3_mass', 'link4_mass', 'link5_mass', 'link6_mass', 'link7_mass', 
    'joint1_damping', 'joint2_damping', 'joint3_damping', 'joint4_damping', 'joint5_damping', 'joint6_damping', 'joint7_damping', 
    'joint1_armature', 'joint2_armature', 'joint3_armature', 'joint4_armature', 'joint5_armature', 'joint6_armature', 'joint7_armature', 
    'actuator_velocity_joint1_kv', 'actuator_velocity_joint2_kv', 'actuator_velocity_joint3_kv', 'actuator_velocity_joint4_kv', 
    'actuator_velocity_joint5_kv', 'actuator_velocity_joint6_kv', 'actuator_velocity_joint7_kv', 
    # 'actuator_position_finger_joint1_kp_1000000', 'actuator_position_finger_joint2_kp_1000000', 
    # 'table_size_0', 'table_size_1', 'table_size_2', 'table_friction_0', 'table_friction_1', 'table_friction_2', 
    'boxobject_size_0', 'boxobject_size_1', 'boxobject_size_2', 
    'boxobject_friction_0', 'boxobject_friction_1', 'boxobject_friction_2', 
    'boxobject_density_1000', 
    # 'pandaik_z_proportional_gain'
    ], 

    'inverteddoublependulumdynamics': [ 
            'damping',
            'gravity',
            'length_1',
            'length_2',
            'density'],

    'halfcheetahdynamics': [ 
        'gravity',
        'bthigh_damping',
        'bshin_damping',
        'bfoot_damping',
        'fthigh_damping',
        'fshin_damping',
        'ffoot_damping',
        'bthigh_stiffness',
        'bshin_stiffness',
        'bfoot_stiffness',
        'fthigh_stiffness',
        'fshin_stiffness',
        'ffoot_stiffness',
    ],
}


HYPER_PARAMS={
    'pandapushik2dsimpledynamics': {'latent_dim': 5, } ,
    'pandapushfkdynamics': {'latent_dim': 5, } ,
    'inverteddoublependulumdynamics': {'latent_dim': 2, } ,
    'halfcheetahdynamics': {'latent_dim': 4, } ,
}
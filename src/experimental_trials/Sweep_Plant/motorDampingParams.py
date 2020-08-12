defaultMotorDamping = 0.00462
scalingFactor = 2
motorDampingParams = {
    "1" : defaultMotorDamping/scalingFactor, # N⋅s⋅m⁻¹
    "2" : defaultMotorDamping, # N⋅s⋅m⁻¹
    "3" : defaultMotorDamping*scalingFactor, # N⋅s⋅m⁻¹
}
# motorDampingParams = {
#     "1" : defaultMotorDamping*(1-scalingFactor), # N⋅s⋅m⁻¹
#     "2" : defaultMotorDamping, # N⋅s⋅m⁻¹
#     "3" : defaultMotorDamping*(1+scalingFactor), # N⋅s⋅m⁻¹
# }

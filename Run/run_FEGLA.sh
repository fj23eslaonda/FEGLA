#!/bin/bash

#--------------------------------
#
# Coquimbo
#
#--------------------------------
python3 RUN_FEGLA_Fr_constant.py --params params_inputs_Coquimbo.json

python3 RUN_FEGLA_Fr_squared.py --params params_inputs_Coquimbo.json

python3 RUN_FEGLA_Fr_linear.py --params params_inputs_Coquimbo.json


#--------------------------------
#
# Viña del Mar
#
#--------------------------------
#python3 RUN_FEGLA_Fr_constant.py --params params_inputs_Vina.json

#python3 RUN_FEGLA_Fr_squared.py --params params_inputs_Vina.json

#python3 RUN_FEGLA_Fr_linear.py --params params_inputs_Vina.json


#--------------------------------
#
# Cartagena
#
#--------------------------------
#python3 RUN_FEGLA_Fr_constant.py --params params_inputs_Cartagena.json

#python3 RUN_FEGLA_Fr_squared.py --params params_inputs_Cartagena.json

#python3 RUN_FEGLA_Fr_linear.py --params params_inputs_Cartagena.json


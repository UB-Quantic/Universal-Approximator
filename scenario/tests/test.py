from Labber import Scenario
from labber.config.lookup import LookUpTable
from Labber import config
import numpy as np

s = Scenario()

# create and add instruments
instr_manual = s.add_instrument('Manual', name='Manual')

# add step items
s.add_step('Manual - Value 1', np.linspace(0, 1, 51))
x = s.add_step('Manual - Value 2', single=0.0)
# create lookup x,y data
xdata = np.linspace(0, 1, 101)
ydata = xdata ** 2
lookup = config.lookup.LookUpTable(xdata, ydata)
# create parameter tied to the lookup table


param = config.step.RelationParameter()
param.variable = 'p1'
param.channel_name = 'Manual - Value 1'
param.use_lookup = False



# param = config.step.RelationParameter()
# param.variable = 'p1'
# param.channel_name = 'Manual - Value 1'
# param.use_lookup = True
# param.lookup = lookup
# enable relation and add parameter to step
x.use_relations = True
x.relation_parameters = [param]
x.equation = 'p1'

s.add_instrument( "Single-Qubit Simulator", name="Qubit Simulator" )

s.add_log("Qubit Simulator - Polarization - Z")

s.log_name = "Test"
s.save("Test")


from psyneulink.components.Function import Linear

from psyneulink.components.mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from psyneulink.components.Process import *

linear_transfer_mechanism = TransferMechanism(function=Linear(slope = 1, intercept = 0))

process_params = {PATHWAY:[linear_transfer_mechanism]}
linear_transfer_process = Process_Base(params = process_params)
linear_transfer_process.execute([1])

process_params = {PATHWAY:[linear_transfer_mechanism]}
linear_transfer_process = Process_Base(params = process_params)

linear_transfer_process.execute([1])
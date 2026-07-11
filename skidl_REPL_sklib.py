from collections import defaultdict
from skidl import Pin, Part, Alias, SchLib, SKIDL, TEMPLATE

from skidl.pin import pin_types

SKIDL_lib_version = '0.0.1'

skidl_REPL = SchLib(tool=SKIDL).add_parts(*[
        Part(**{ 'name':'OPA340NA', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'OPA340NA'}), 'ref_prefix':'U', 'fplist':['Package_TO_SOT_SMD:SOT-23-5', 'Package_TO_SOT_SMD:SOT-23-5'], 'footprint':'Package_TO_SOT_SMD:SOT-23-5', 'keywords':'single opamp', 'description':'', 'datasheet':'http://www.ti.com/lit/ds/symlink/opa340.pdf', 'pins':[
            Pin(num='2',name='V-',func=pin_types.PWRIN),
            Pin(num='5',name='V+',func=pin_types.PWRIN),
            Pin(num='1',name='~',func=pin_types.OUTPUT,unit=1),
            Pin(num='3',name='+',func=pin_types.INPUT,unit=1),
            Pin(num='4',name='-',func=pin_types.INPUT,unit=1)], 'unit_defs':[] }),
        Part(**{ 'name':'R', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'R'}), 'ref_prefix':'R', 'fplist':[''], 'footprint':'', 'keywords':'R res resistor', 'description':'', 'datasheet':'~', 'pins':[
            Pin(num='1',name='~',func=pin_types.PASSIVE,unit=1),
            Pin(num='2',name='~',func=pin_types.PASSIVE,unit=1)], 'unit_defs':[] }),
        Part(**{ 'name':'C', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'C'}), 'ref_prefix':'C', 'fplist':[''], 'footprint':'', 'keywords':'cap capacitor', 'description':'', 'datasheet':'~', 'pins':[
            Pin(num='1',name='~',func=pin_types.PASSIVE,unit=1),
            Pin(num='2',name='~',func=pin_types.PASSIVE,unit=1)], 'unit_defs':[] })])
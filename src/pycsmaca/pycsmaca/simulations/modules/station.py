from pydesim import Model

from pycsmaca.simulations.modules import Sink, NetworkService, NetworkSwitch


class Station(Model):
    def __init__(self, sim, source, interfaces):
        super().__init__(sim)

        # Creating missing modules:
        sink = Sink(sim)
        network_service = NetworkService(sim)
        switch = NetworkSwitch(sim)

        # Registering children:
        if source is not None:
            self.children['source'] = source
        self.children['sink'] = sink
        self.children['network_service'] = network_service
        self.children['switch'] = switch
        self.children['interfaces'] = interfaces

        # Connecting modules:
        if source is not None:
            source.connections.set('network', network_service, rname='source')
        sink.connections.set('network', network_service, rname='sink')
        network_service.connections.set('network', switch, rname='user')
        for i, iface in enumerate(interfaces):
            switch.connections.set(f'if{i}', iface, rname='user')
    
    @property
    def source(self):
        return self.children['source'] if 'source' in self.children else None
    
    @property
    def sink(self):
        return self.children['sink']
    
    @property
    def network_service(self):
        return self.children['network_service']
    
    @property
    def switch(self):
        return self.children['switch']
    
    @property
    def interfaces(self):
        return self.children['interfaces']

    def get_interface_by_address(self, address):
        for iface in self.children['interfaces']:
            if iface.address == address:
                return iface

    def get_interface_to(self, remote_sta):
        #
        # If remote_sta is found in switching table, return the interface
        # described by it:
        #
        for remote_address in (nif.address for nif in remote_sta.interfaces):
            if remote_address in self.switch.table:
                conn_name = self.switch.table[remote_address].connection
                return self.switch.connections[conn_name].module
            for link in self.switch.table.as_dict().values():
                if link[1] == remote_address:
                    conn_name = link[0]
                    return self.switch.connections[conn_name].module

        #
        # Otherwise, inspect neighbours:
        #
        def test_parent_is_remote_sta(module):
            if module.parent is None:
                return False
            return module.parent == remote_sta or test_parent_is_remote_sta(
                module.parent)
        for iface in self.interfaces:
            if 'wire' in iface.connections:
                peer = iface.connections['wire'].module
                if test_parent_is_remote_sta(peer):
                    return iface

        #
        # If neither found, return None:
        #
        return None

    def get_switch_connection_for(self, iface):
        for conn_name in self.switch.connections.names():
            conn = self.switch.connections[conn_name]
            if conn.module == iface:
                return conn
        return None

    def __str__(self):
        suffix = ''
        if self.parent:
            prefix = f'{self.parent}.'
            # TODO: this is awful hardcode! Need some way to address self index.
            if ('stations' in self.parent.children and
                    self in self.parent.children['stations']):
                suffix = f'[{self.parent.children["stations"].index(self)}]'
        else:
            prefix = ''
        return f'{prefix}Station{suffix}'

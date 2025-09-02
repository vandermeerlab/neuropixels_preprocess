# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:30:21 2020

@author: Julie
"""
# import from plugins/custom_columns.py
from phy import IPlugin, connect
"""only keep some columns in cluster view."""

class removeColumnsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_controller_ready(sender):
            """edit controller.supervisor.columns below to only include metrics you want to display in phy"""
            controller.supervisor.columns = ['id', 'n_spikes', 'depth',
                                             'frac_RPVs', 'percentage_spikes_missing', 'fr', 'amp',
                                              'KSLabel', 'bc_unitType'] 

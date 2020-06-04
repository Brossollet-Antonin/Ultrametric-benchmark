fs.cf_stats['stat_list'][0]['avg_cf'] = {k: v/fs.cf_stats['stat_list'][1]['avg_cf'][k] for k,v in fs.cf_stats['stat_list'][0]['avg_cf'].items() if fs.cf_stats['stat_list'][1]['avg_cf'][k]>0}
fs.cf_stats['stat_list'][0]['avg_cf_std'] = {k: v/fs.cf_stats['stat_list'][1]['avg_cf'][k] for k,v in fs.cf_stats['stat_list'][0]['avg_cf_std'].items() if fs.cf_stats['stat_list'][1]['avg_cf'][k]>0}

fs.cf_stats['stat_list'][2]['avg_cf'] = {k: v/fs.cf_stats['stat_list'][3]['avg_cf'][k] for k,v in fs.cf_stats['stat_list'][2]['avg_cf'].items() if fs.cf_stats['stat_list'][3]['avg_cf'][k]>0}
fs.cf_stats['stat_list'][2]['avg_cf_std'] = {k: v/fs.cf_stats['stat_list'][3]['avg_cf'][k] for k,v in fs.cf_stats['stat_list'][2]['avg_cf_std'].items() if fs.cf_stats['stat_list'][3]['avg_cf'][k]>0}

del fs.cf_stats['stat_list'][3]
del fs.cf_stats['stat_list'][1]
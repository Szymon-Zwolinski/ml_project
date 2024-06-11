data = [
    ['<25', 'low', 'good'], ['<25', 'low', 'bad'], ['<25', 'low', 'good'], ['<25', 'low', 'good'], ['<25', 'low', 'good'], 
    ['<25', 'low', 'bad'], ['<25', 'low', 'good'], ['<25', 'low', 'good'], ['<25', 'low', 'good'], ['<25', 'low', 'bad'], 
    ['<25', 'low', 'good'], ['<25', 'low', 'bad'], ['<25', 'low', 'good'], ['<25', 'low', 'good'], ['<25', 'low', 'good'], 
    ['<25', 'low', 'good'], ['<25', 'low', 'bad'], ['<25', 'low', 'good'], ['<25', 'low', 'good'], ['<25', 'low', 'bad'], 
    ['<25', 'medium', 'good'], ['<25', 'medium', 'bad'], ['<25', 'medium', 'good'], ['<25', 'medium', 'good'], ['<25', 'medium', 'good'], 
    ['<25', 'medium', 'bad'], ['<25', 'medium', 'good'], ['<25', 'medium', 'good'], ['<25', 'medium', 'bad'], ['<25', 'medium', 'good'], 
    ['<25', 'medium', 'good'], ['<25', 'medium', 'bad'], ['<25', 'medium', 'good'], ['<25', 'medium', 'good'], ['<25', 'medium', 'bad'], 
    ['<25', 'medium', 'good'], ['<25', 'medium', 'good'], ['<25', 'medium', 'good'], ['<25', 'medium', 'good'], ['<25', 'high', 'good'], 
    ['<25', 'high', 'good'], ['<25', 'high', 'bad'], ['<25', 'high', 'good'], ['<25', 'high', 'good'], ['<25', 'high', 'bad'], 
    ['<25', 'high', 'good'], ['<25', 'high', 'good'], ['<25', 'high', 'good'], ['<25', 'high', 'good'], ['<25', 'high', 'bad'], 
    ['25-35', 'low', 'good'], ['25-35', 'low', 'bad'], ['25-35', 'low', 'good'], ['25-35', 'low', 'good'], ['25-35', 'low', 'good'], 
    ['25-35', 'low', 'bad'], ['25-35', 'low', 'good'], ['25-35', 'low', 'bad'], ['25-35', 'low', 'good'], ['25-35', 'low', 'good'], 
    ['25-35', 'low', 'good'], ['25-35', 'low', 'bad'], ['25-35', 'low', 'good'], ['25-35', 'low', 'good'], ['25-35', 'low', 'bad'], 
    ['25-35', 'low', 'good'], ['25-35', 'low', 'good'], ['25-35', 'low', 'good'], ['25-35', 'low', 'good'], ['25-35', 'medium', 'good'], 
    ['25-35', 'medium', 'good'], ['25-35', 'medium', 'bad'], ['25-35', 'medium', 'good'], ['25-35', 'medium', 'good'], ['25-35', 'medium', 'bad'], 
    ['25-35', 'medium', 'good'], ['25-35', 'medium', 'good'], ['25-35', 'medium', 'good'], ['25-35', 'medium', 'bad'], ['25-35', 'medium', 'good'], 
    ['25-35', 'medium', 'good'], ['25-35', 'medium', 'bad'], ['25-35', 'medium', 'good'], ['25-35', 'medium', 'good'], ['25-35', 'medium', 'good'], 
    ['25-35', 'medium', 'good'], ['25-35', 'high', 'good'], ['25-35', 'high', 'bad'], ['25-35', 'high', 'good'], ['25-35', 'high', 'good'], 
    ['25-35', 'high', 'good'], ['25-35', 'high', 'bad'], ['25-35', 'high', 'good'], ['25-35', 'high', 'good'], ['25-35', 'high', 'good'], 
    ['25-35', 'high', 'good'], ['35-45', 'low', 'good'], ['35-45', 'low', 'bad'], ['35-45', 'low', 'good'], ['35-45', 'low', 'bad'], 
    ['35-45', 'low', 'good'], ['35-45', 'low', 'bad'], ['35-45', 'low', 'good'], ['35-45', 'low', 'bad'], ['35-45', 'low', 'good'], 
    ['35-45', 'low', 'bad'], ['35-45', 'low', 'good'], ['35-45', 'low', 'bad'], ['35-45', 'low', 'good'], ['35-45', 'low', 'bad'], 
    ['35-45', 'low', 'good'], ['35-45', 'low', 'bad'], ['35-45', 'medium', 'good'], ['35-45', 'medium', 'bad'], ['35-45', 'medium', 'good'], 
    ['35-45', 'medium', 'bad'], ['35-45', 'medium', 'good'], ['35-45', 'medium', 'bad'], ['35-45', 'medium', 'good'], ['35-45', 'medium', 'bad'], 
    ['35-45', 'medium', 'good'], ['35-45', 'medium', 'bad'], ['35-45', 'medium', 'good'], ['35-45', 'medium', 'bad'], ['35-45', 'medium', 'good'], 
    ['35-45', 'medium', 'bad'], ['35-45', 'medium', 'good'], ['35-45', 'medium', 'bad'], ['35-45', 'high', 'good'], ['35-45', 'high', 'bad'], 
    ['35-45', 'high', 'good'], ['35-45', 'high', 'bad'], ['35-45', 'high', 'good'], ['35-45', 'high', 'bad'], ['35-45', 'high', 'good'], 
    ['35-45', 'high', 'bad'], ['35-45', 'high', 'good'], ['35-45', 'high', 'bad'], ['35-45', 'high', 'good'], ['35-45', 'high', 'bad'], 
    ['35-45', 'high', 'good'], ['35-45', 'high', 'bad'], ['>45', 'low', 'good'], ['>45', 'low', 'bad'], ['>45', 'low', 'good'], 
    ['>45', 'low', 'bad'], ['>45', 'low', 'good'], ['>45', 'low', 'bad'], ['>45', 'low', 'good'], ['>45', 'low', 'bad'], 
    ['>45', 'low', 'good'], ['>45', 'low', 'bad'], ['>45', 'low', 'good'], ['>45', 'low', 'bad'], ['>45', 'low', 'good'], 
    ['>45', 'low', 'bad'], ['>45', 'medium', 'good'], ['>45', 'medium', 'bad'], ['>45', 'medium', 'good'], ['>45', 'medium', 'bad'], 
    ['>45', 'medium', 'good'], ['>45', 'medium', 'bad'], ['>45', 'medium', 'good'], ['>45', 'medium', 'bad'], ['>45', 'medium', 'good'], 
    ['>45', 'medium', 'bad'], ['>45', 'medium', 'good'], ['>45', 'medium', 'bad'], ['>45', 'medium', 'good'], ['>45', 'medium', 'bad'], 
    ['>45', 'high', 'good'], ['>45', 'high', 'bad'], ['>45', 'high', 'good'], ['>45', 'high', 'bad'], ['>45', 'high', 'good'], 
    ['>45', 'high', 'bad'], ['>45', 'high', 'good'], ['>45', 'high', 'bad'], ['>45', 'high', 'good'], ['>45', 'high', 'bad'], 
    ['>45', 'high', 'good'], ['>45', 'high', 'bad']
]
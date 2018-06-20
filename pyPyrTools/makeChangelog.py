#!/usr/bin/python
# script makes a change log txt file from git logs
# usage: makeChangelog.py > CHANGELOG.txt

import subprocess

atags = subprocess.getoutput('git tag -n')
atags = atags.split('\n');
tags = subprocess.getoutput('git tag -l')
tags = tags.split('\n');

#for t in range(0, len(tags)-1):
for t in range(len(tags)-1, -1, -1):
    if t == 0:
        #print "*** " + atags[t+1]
        print("*** " + atags[t])
    else:
        #print '\n\n*** ' + atags[t+1]
        print('\n\n*** ' + atags[t])
    #commandStr = 'git log %s..%s --pretty=%s' % (tags[t], tags[t+1], '%s')
    commandStr = 'git log %s..%s --pretty=%s' % (tags[t-1], tags[t], '%s')
    changes = subprocess.getoutput(commandStr)
    changes = changes.split('\n')
    #changes = changes[::-1]
    for line in changes:
        print('  + ' + line)

#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import argparse
import logging
import datetime
from copy import deepcopy
from hashlib import sha1

import signac

logging.basicConfig(filename='init.log', filemode='w', level=logging.INFO)

'''
-----------------------------
NEW 3 terminal group chemistries
-----------------------------
NOTE: used to be 5 new chemistries, but there have been issues
with the stability of these systems, or missing ff parameters.

The top layer will be one of the 16 original terminal group chemistries,
the bottom layer the 3 NEW chemistries
'''
terminal_groups_A = ['acetyl', 'amino', 'carboxyl', 'cyano', 'cyclopropyl',
                     'ethylene', 'fluorophenyl', 'hydroxyl', 'isopropyl',
                     'methoxy', 'methyl', 'nitro', 'nitrophenyl',
                     'perfluoromethyl', 'phenyl', 'pyrrole']

terminal_groups_new_B = ['toluene', 'phenol', 'difluoromethyl']



# Initialize the project
def main(args, random_seed):
    project = signac.init_project("TerminalGroupScreeningNewWithOld")
    logging.info("Init begin: {}".format(datetime.datetime.today()))
    logging.info("Initialized project name")
    statepoints = list()
    # generate the new top and bottom monolayers
    for replication_index in range(args.num_replicas):
        for terminal_group_a in terminal_groups_A:
            for terminal_group_b in terminal_groups_new_B:
                if terminal_group_a != terminal_group_b:
                    the_statepoint = dict(
                            # Carbon backbone length
                            chainlength = 17,
                            # Number of monolayer chains
                            n = 100,
                            # Random seed
                            seed = random_seed*(replication_index + 1),
                            # Terminal group chemistries
                            terminal_groups = tuple(sorted((terminal_group_a,
                                terminal_group_b))))
                project.open_job(statepoint=the_statepoint).init()
                statepoints.append(the_statepoint)
                logging.info(msg="At the statepoint: {}".format(the_statepoint))
    # write statepoints to signac statepoint file
    project.write_statepoints(statepoints=statepoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Initialize the data space.")
    parser.add_argument(
        'random',
        type=str,
        help="A string to generate a random seed.")
    parser.add_argument(
        '-n', '--num-replicas',
        type=int,
        default=1,
        help="Initialize multiple replications.")
    parser.add_argument(
        '-c', '--chain-length',
        type=int,
        default=17,
        help="Backbone length of chain.")
    args = parser.parse_args()

    # Generate an integer from the random str.
    try:
        random_seed = int(args.random)
    except ValueError:
        random_seed = int(sha1(args.random.encode()).hexdigest(), 16) % (10 ** 8)

    logging.info("Params:\
                random: {}\
                num-replicas: {}\
                chain_length: {}".format(
                    random_seed,
                    args.num_replicas,
                    args.chain_length
                    ))

    main(args, random_seed)

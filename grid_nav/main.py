#!/usr/bin/env python3

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Grid Navigation')
    parser.add_argument('--world', default='grid1', help='World file to load (default: grid1)')
    parser.add_argument('--agent', choices=['human', 'A*'], default='human', 
                        help='Agent type to use (default: human)')
    args = parser.parse_args()
    
    # Construct the path to the world file
    world_file = f'worlds/{args.world}.txt'
    
    # Check if file exists
    if not os.path.exists(world_file):
        print(f"Error: World file '{world_file}' not found")
        return
   
    # Display
    print(f"World: {args.world}")
    print(f"Agent: {args.agent}")
    print("-" * 20)
    with open(world_file, 'r') as f:
        for line in f:
            print(line.rstrip())
    print("-" * 20)


if __name__ == '__main__':
    main()


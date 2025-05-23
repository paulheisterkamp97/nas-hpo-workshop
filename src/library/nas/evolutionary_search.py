"""
Evolutionary Search for Neural Architecture Search (NAS)

This module implements an evolutionary algorithm for NAS, including:
1. Simple Genetic Algorithm (GA)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from copy import deepcopy

from .search_space import SearchSpace
from src.library.nas.utils_2 import get_dataset, train_model, evaluate_model, get_best_torch_device

class EvolutionarySearch:
    """
    Evolutionary Search for Neural Architecture Search
    
    This class implements a Simple Genetic Algorithm (GA): Optimizes for a single objective (accuracy)
    as an evolutionary algorithm for NAS:
    """
    def __init__(self, search_space, method='simple_ga', population_size=20, num_generations=3,
                 mutation_rate=0.2, crossover_rate=0.5, tournament_size=3, epochs_per_model=5, 
                 batch_size=64, device=None, small_subset=True):
        """
        Initialize the Evolutionary Search
        
        Args:
            search_space: SearchSpace object defining the architecture search space
            method: Evolutionary method to use ('simple_ga' or later 'nsga_ii (not implemented in this workshop)')
            population_size: Size of the population
            num_generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for selection
            epochs_per_model: Number of epochs to train each model
            batch_size: Batch size for training
            device: Device to use for training (if None, use best available)
            small_subset: Whether to use a small subset of data for faster evaluation
        """
        self.search_space = search_space
        self.method = method
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.epochs_per_model = epochs_per_model
        self.batch_size = batch_size
        self.small_subset = small_subset
        
        # Automatically select the best available device (GPU/CPU)
        if device is None:
            self.device = get_best_torch_device()
        else:
            self.device = device
            
        # Load datasets for training and evaluation
        self.train_loader, self.val_loader, self.test_loader = get_dataset(
            batch_size=batch_size, small_subset=small_subset
        )
        
        # Storage for search results
        self.population = []
        self.fitness_history = []
        self.best_architecture = None
        self.best_accuracy = 0
        self.best_model = None
    
    def search(self, verbose=True, use_simple_sampling=False):
        """
        Perform architecture search using the selected evolutionary method
        
        Args:
            verbose: Whether to print progress
            use_simple_sampling: Whether to use simple sampling instead of evolution (for demonstration)
            
        Returns:
            Best architecture found and its corresponding model
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting {self.method.replace('_', ' ').upper()} Search with {self.population_size} individuals and {self.num_generations} generations...")

        self.search_simple_ga(verbose)
        
        end_time = time.time()
        search_time = (end_time - start_time) / 60  # in minutes
        
        if verbose:
            print(f"Evolutionary Search completed in {search_time:.2f} minutes")
            print(f"Best architecture accuracy: {self.best_accuracy:.2f}%")
        
        return self.best_architecture, self.best_model
    
    def initialize_population(self):
        """
        Initialize the population with random architectures
        
        This creates a diverse initial population by sampling random
        architectures from the search space.
        """
        self.population = []
        
        for _ in range(self.population_size):
            architecture = self.search_space.sample_random_architecture()
            individual = {
                'architecture': architecture,
                'accuracy': None,
                'ms_per_image': None,
                'model': None
            }
            self.population.append(individual)
    
    def evaluate_population(self, individuals=None, verbose=True):
        """
        Evaluate the fitness of individuals in the population
        
        This trains and evaluates each model to determine its fitness
        (accuracy and inference time).
        
        Args:
            individuals: List of individuals to evaluate (if None, evaluate the entire population)
            verbose: Whether to print progress
        """
        if individuals is None:
            individuals = self.population
            
        if verbose:
            print(f"Evaluating {len(individuals)} individuals...")
            iterator = tqdm(individuals)
        else:
            iterator = individuals
        
        for individual in iterator:
            if individual['accuracy'] is not None:
                # Skip already evaluated individuals
                continue
                
            # Step 1: Build model from architecture
            model = self.search_space.build_model_from_architecture(individual['architecture'])
            
            # Step 2: Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Step 3: Train the model
            history = train_model(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                num_epochs=self.epochs_per_model,
                verbose=False
            )
            
            # Step 4: Evaluate the model
            eval_results = evaluate_model(
                model=model,
                test_loader=self.test_loader,
                device=self.device,
                verbose=False
            )
            
            # Step 5: Update individual with fitness values
            individual['accuracy'] = eval_results['accuracy']
            individual['ms_per_image'] = eval_results['ms_per_image']
            individual['model'] = model
            
            # Step 6: Update best architecture if needed
            if individual['accuracy'] > self.best_accuracy:
                self.best_architecture = individual['architecture']
                self.best_accuracy = individual['accuracy']
                self.best_model = model
                
                if verbose:
                    tqdm.write(f"New best architecture found! Accuracy: {self.best_accuracy:.2f}%")

    
    def selection_tournament(self, k=3):
        """
        Tournament selection
        
        Randomly selects k individuals and returns the best one.
        
        Args:
            k: Tournament size
            
        Returns:
            Selected individual
        """
        # Randomly select k individuals
        tournament = random.sample(self.population, k)
        
        # Return the best individual in the tournament
        best_individual = None
        best_accuracy = -1
        
        for individual in tournament:
            if individual['accuracy'] > best_accuracy:
                best_individual = individual
                best_accuracy = individual['accuracy']
                
        return best_individual
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents
        
        Creates a new child architecture by combining features from both parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child architecture
        """
        if random.random() < self.crossover_rate:
            # Perform crossover
            child_arch = self.search_space.crossover(parent1['architecture'], parent2['architecture'])
        else:
            # No crossover, just clone one parent
            if random.random() < 0.5:
                child_arch = deepcopy(parent1['architecture'])
            else:
                child_arch = deepcopy(parent2['architecture'])
            
        return {
            'architecture': child_arch,
            'accuracy': None,
            'ms_per_image': None,
            'model': None,
            'rank': None,
            'crowding_distance': None,
            'dominated_count': None,
            'dominated_solutions': None,
        }
    
    def mutation(self, individual):
        """
        Perform mutation on an individual
        
        Introduces random changes to the architecture to maintain diversity.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        if random.random() < self.mutation_rate:
            # Perform mutation
            mutated_arch = self.search_space.mutate_architecture(individual['architecture'])
            individual['architecture'] = mutated_arch
            individual['accuracy'] = None  # Reset fitness values
            individual['ms_per_image'] = None
            individual['model'] = None
            
        return individual
    
    def search_simple_ga(self, verbose=True):
        """
        Perform architecture search using simple genetic algorithm
        
        This implements a standard genetic algorithm with:
        - Tournament selection
        - Crossover and mutation
        - Elitism (preserving the best individual)
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Best architecture found
        """
        # Step 1: Initialize population
        self.initialize_population()
        
        # Step 2: Evaluate initial population
        self.evaluate_population(verbose=verbose)
        
        # Step 3: Store initial fitness
        avg_fitness = sum(ind['accuracy'] for ind in self.population) / len(self.population)
        best_fitness = max(ind['accuracy'] for ind in self.population)
        self.fitness_history.append((avg_fitness, best_fitness))
        
        if verbose:
            print(f"Generation 0: Avg Fitness = {avg_fitness:.2f}%, Best Fitness = {best_fitness:.2f}%")
        
        # Step 4: Evolution loop
        for generation in range(1, self.num_generations + 1):
            if verbose:
                print(f"\nGeneration {generation}/{self.num_generations}")
            
            # Step 4.1: Create new population
            new_population = []
            
            # Step 4.2: Elitism - keep the best individual
            best_individual = None
            best_accuracy = -1
            
            for individual in self.population:
                if individual['accuracy'] > best_accuracy:
                    best_individual = individual
                    best_accuracy = individual['accuracy']
                    
            new_population.append(deepcopy(best_individual))
            
            # Step 4.3: Create offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.selection_tournament(self.tournament_size)
                parent2 = self.selection_tournament(self.tournament_size)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutation(child)
                
                new_population.append(child)
            
            # Step 4.4: Replace population
            self.population = new_population
            
            # Step 4.5: Evaluate new population
            self.evaluate_population(verbose=verbose)
            
            # Step 4.6: Store fitness history
            avg_fitness = sum(ind['accuracy'] for ind in self.population) / len(self.population)
            best_fitness = max(ind['accuracy'] for ind in self.population)
            self.fitness_history.append((avg_fitness, best_fitness))
            
            if verbose:
                print(f"Generation {generation}: Avg Fitness = {avg_fitness:.2f}%, Best Fitness = {best_fitness:.2f}%")
        
        return self.best_architecture, self.best_model
    
    def plot_fitness_history(self):
        """
        Plot the fitness history of the evolutionary search
        """
        if not self.fitness_history:
            print("No fitness history to plot. Run search() first.")
            return
        
        avg_fitness = [h[0] for h in self.fitness_history]
        best_fitness = [h[1] for h in self.fitness_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(avg_fitness, label='Average Fitness')
        plt.plot(best_fitness, label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{self.method.replace("_", " ").upper()} - Fitness History')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    

    def get_best_architecture_summary(self):
        """
        Get a human-readable summary of the best architecture found
        """
        if self.best_architecture is None:
            return "No architecture found. Run search() first."
        
        arch = self.best_architecture
        
        summary = "Best Architecture Summary:\n"
        summary += f"Accuracy: {self.best_accuracy:.2f}%\n\n"
        
        summary += "Convolutional Layers:\n"
        for i, layer in enumerate(arch['conv_layers']):
            summary += f"Layer {i+1}:\n"
            summary += f"  Operation: {layer['op']}\n"
            summary += f"  Channels: {layer['in_channels']} -> {layer['out_channels']}\n"
            summary += f"  Activation: {layer['activation']}\n"
            summary += f"  Pooling: {layer['pooling'] if layer['use_pooling'] else 'None'}\n"
        
        summary += "\nFully Connected Layers:\n"
        for i, layer in enumerate(arch['fc_layers']):
            summary += f"Layer {i+1}:\n"
            summary += f"  Size: {layer['in_features']} -> {layer['out_features']}\n"
            summary += f"  Activation: {layer['activation'] if layer['activation'] else 'None'}\n"
            summary += f"  Dropout: {layer['dropout_rate']}\n"
        
        return summary

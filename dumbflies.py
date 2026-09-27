import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
import time
from dataclasses import dataclass, field
from collections import deque
import logging
import sys
from typing import Dict, Tuple, List, Optional
import random
from scipy.ndimage import gaussian_filter
import matplotlib
import csv
import os
import json

# Use TkAgg backend for Matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# Configuration and Data Classes
# -----------------------------

@dataclass
class BugConfig:
    # Vision settings
    vision_cone_angle: float = np.pi / 3  # 60 degrees in radians
    vision_cone_length: int = 150
    vision_width: int = 32
    vision_height: int = 32

    # Camera settings
    camera_index: int = 0       # Default camera index
    camera_width: int = 1280
    camera_height: int = 720

    # Movement settings
    max_speed: float = 15.0     # Increased for faster movement
    turn_speed: float = 0.2     # Reduced for smoother turns
    momentum: float = 0.5       # Reduced for more responsive changes
    random_movement: float = 0.2 # Random movement factor
    hover_amplitude: float = 2.0 # Hover movement

    # Population settings
    initial_population: int = 15
    max_population: int = 100   # Increased max population
    min_population: int = 5     # Minimum population

    # Energy settings
    energy_decay: float = 0.05  # Reduced for longer life
    mating_threshold: float = 80.0
    initial_energy: float = 100.0
    reproduction_cost: float = 40.0

    # Spacing settings
    repulsion_distance: float = 30.0   # Increased from 5
    repulsion_force: float = 2.0        # Increased from 0.70
    personal_space: float = 15.0        # Increased from 3

    # Bug detection settings
    bug_visibility_size: int = 5        # Size of bug markers in vision
    bug_detection_range: float = 100.0  # How far away bugs can see each other

    # Visual effects
    birth_marker_duration: int = 30     # How many frames to show birth marker
    death_marker_duration: int = 50     # How many frames to show death marker

    # Genetic settings
    mutation_rate: float = 0.05  # Probability of mutation per trait

@dataclass
class VisualEffect:
    x: float
    y: float
    duration: int
    effect_type: str  # 'birth' or 'death'
    age: int = 0

    def update(self) -> bool:
        self.age += 1
        return self.age < self.duration

    def draw(self, frame):
        if self.effect_type == 'birth':
            # Draw expanding circle for birth
            radius = int(self.age / 2)
            color = (0, 255, 0)  # Green for birth
            cv2.circle(frame, (int(self.x), int(self.y)), radius, color, 1)
        elif self.effect_type == 'death':
            # Draw small black dot for death
            cv2.circle(frame, (int(self.x), int(self.y)), 2, (0, 0, 0), -1)

# -----------------------------
# Thinking Field Class
# -----------------------------

class ThinkingField:
    def __init__(self, field_size=32, pattern_threshold=0.5, stability=5.0, decay_rate=0.01, seed=None):
        self.field_size = field_size
        if seed is not None:
            np.random.seed(seed)
        self.field = np.random.rand(field_size, field_size) * 0.1  # Start with low energy
        self.pattern_threshold = pattern_threshold
        self.stability = stability
        self.decay_rate = decay_rate
        self.patterns = []

    def apply_input(self, input_array):
        """
        Integrate vision input into the field.
        """
        # Normalize input
        input_normalized = (input_array - np.min(input_array)) / (np.max(input_array) - np.min(input_array) + 1e-7)
        # Reshape to match field size
        input_resized = cv2.resize(input_normalized, (self.field_size, self.field_size))
        # Integrate input into the field
        self.field += input_resized * 0.1  # Scaling factor for influence

    def evolve(self, drift=0.01, quantum_kick=0.05):
        """
        Evolve the field by applying drift and random perturbations.
        """
        # Apply drift towards 0.5 to stabilize
        self.field += drift * (0.5 - self.field)
        # Apply random perturbations (quantum kicks)
        random_matrix = np.random.randn(self.field_size, self.field_size) * quantum_kick
        self.field += self.apply_spatial_correlation(random_matrix)
        # Apply decay
        self.field *= (1 - self.decay_rate)
        # Clamp field values
        self.field = np.clip(self.field, 0, 1)

    def apply_spatial_correlation(self, noise_array):
        """
        Apply Gaussian blur to simulate spatial correlation.
        """
        return gaussian_filter(noise_array, sigma=1)

    def detect_patterns(self):
        """
        Detect stable patterns in the field.
        """
        stable_patterns = []
        for i in range(self.field_size):
            for j in range(self.field_size):
                if self.field[i, j] > self.pattern_threshold:
                    stable_patterns.append((i, j))
        self.patterns = stable_patterns
        return stable_patterns

    def generate_decision(self):
        """
        Generate a decision based on stable patterns.
        """
        if not self.patterns:
            return (0.0, 0.0, False, False)  # No movement, no actions

        # Calculate multiple decision metrics based on patterns
        avg_i = np.mean([p[0] for p in self.patterns])
        avg_j = np.mean([p[1] for p in self.patterns])

        # Determine movement direction based on pattern centroid
        direction_x = (avg_j - self.field_size / 2) / (self.field_size / 2)
        direction_y = (avg_i - self.field_size / 2) / (self.field_size / 2)

        # Calculate pattern density for actions
        pattern_density = len(self.patterns) / (self.field_size * self.field_size)

        # Define thresholds for actions
        eating_threshold = 0.1
        mating_threshold = 0.2

        eating = pattern_density > eating_threshold
        mating = pattern_density > mating_threshold

        return (direction_x, direction_y, eating, mating)

    def associate_patterns(self, source_patterns: List[Tuple[int, int]]):
        """
        Associate patterns from another field to influence current field.
        """
        for pattern in source_patterns:
            # Simple association: reinforce nearby areas
            i, j = pattern
            # Apply a small reinforcement around the pattern
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.field_size and 0 <= nj < self.field_size:
                        self.field[ni, nj] += 0.02  # Reinforcement factor
        # Ensure field values are clamped
        self.field = np.clip(self.field, 0, 1)

    def blend_fields(self, other_field: 'ThinkingField'):
        """
        Blend this field with another field to create a mixed field.
        """
        blended_field = (self.field + other_field.field) / 2
        return blended_field

    def copy_field(self):
        """
        Return a copy of the field.
        """
        return self.field.copy()

# -----------------------------
# FlyVisuals Class
# -----------------------------

class FlyVisuals:
    """Enhanced fly visualization with more realistic effects"""
    def __init__(self):
        self.birth_particles = []  # Store birth effect particles
        self.mating_glow_phase = 0.0  # For smooth mating glow animation

    def is_within_frame(self, x, y, frame) -> bool:
        """Check if (x, y) is within the frame boundaries."""
        height, width = frame.shape[:2]
        return 0 <= x < width and 0 <= y < height

    def draw_fly(self, frame, x, y, angle, is_mating, size, shade):
        """Enhanced fly drawing with more natural animations"""
        # Convert coordinates to integers
        x, y = int(x), int(y)

        # Wing animation with varying speed based on movement
        wing_phase = (time.time() % (2 * np.pi)) * 3  # Faster wing movement
        wing_amplitude = 0.8 if is_mating else 0.5  # Wings spread wider during mating

        # Body parameters
        body_length = size * 1.2  # Slightly longer body
        body_width = max(2, int(size / 1.8))  # Thinner body

        # Calculate body segments (more realistic fly shape)
        segments = 3
        for i in range(segments):
            segment_ratio = i / segments
            segment_width = max(1, int(body_width * (1.0 - segment_ratio * 0.3)))  # Ensure width >=1
            segment_x = x - body_length * np.cos(angle) * segment_ratio
            segment_y = y - body_length * np.sin(angle) * segment_ratio

            # Determine color with more natural shading
            base_shade = int(shade * 255)
            if is_mating:
                # Subtle pulsing glow during mating
                glow = np.sin(self.mating_glow_phase) * 0.3 + 0.7
                r = int(min(255, base_shade * 1.2))
                g = int(base_shade * 0.8 * glow)
                b = int(min(255, base_shade * 1.1))
                color = (r, g, b)
                self.mating_glow_phase += 0.1
            else:
                # Natural gray with slight iridescence
                color = (
                    int(base_shade * 0.9),
                    int(base_shade * 1.0),
                    int(base_shade * 1.1)
                )

            if self.is_within_frame(segment_x, segment_y, frame):
                cv2.circle(frame, 
                          (int(segment_x), int(segment_y)), 
                          segment_width,
                          color, 
                          -1)

        # Draw wings with transparency effect
        for wing_side in [-1, 1]:
            wing_angle = angle + np.pi/2 * wing_side + np.sin(wing_phase) * wing_amplitude
            wing_length = size * 1.5

            # Calculate wing points for membrane effect
            wing_points = np.array([
                [x, y],
                [int(x + wing_length * 0.7 * np.cos(wing_angle - 0.2)),
                 int(y + wing_length * 0.7 * np.sin(wing_angle - 0.2))],
                [int(x + wing_length * np.cos(wing_angle)),
                 int(y + wing_length * np.sin(wing_angle))],
                [int(x + wing_length * 0.7 * np.cos(wing_angle + 0.2)),
                 int(y + wing_length * 0.7 * np.sin(wing_angle + 0.2))]
            ], dtype=np.int32)

            # Draw semi-transparent wing membrane
            wing_color = (240, 240, 240)
            cv2.fillPoly(frame, [wing_points], wing_color)

            # Draw wing veins
            vein_end_x = int(x + wing_length * np.cos(wing_angle))
            vein_end_y = int(y + wing_length * np.sin(wing_angle))
            if self.is_within_frame(vein_end_x, vein_end_y, frame):
                cv2.line(frame, (x, y), 
                        (vein_end_x, vein_end_y),
                        (180, 180, 180), 1)

        # Draw antennae with segments
        antenna_segments = 3
        for ant_side in [-1, 1]:
            prev_x, prev_y = x, y
            ant_angle = angle + np.pi/6 * ant_side
            segment_length = size * 0.3

            for seg in range(antenna_segments):
                # Add slight curve to antennae
                ant_angle += np.sin(time.time() * 2 + seg) * 0.1
                end_x = int(prev_x + segment_length * np.cos(ant_angle))
                end_y = int(prev_y + segment_length * np.sin(ant_angle))
                if self.is_within_frame(end_x, end_y, frame):
                    cv2.line(frame, (prev_x, prev_y), (end_x, end_y), (0, 0, 0), 1)
                    prev_x, prev_y = end_x, end_y

# -----------------------------
# EnhancedBug Class
# -----------------------------

class EnhancedBug:
    def __init__(self, x: int, y: int, config: BugConfig, traits: Optional[Dict] = None, seed=None):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * np.pi)
        self.config = config

        # Initialize Thinking Fields with optional seed
        self.perception_field = ThinkingField(field_size=32, pattern_threshold=0.6, stability=5.0, seed=seed)
        self.memory_field = ThinkingField(field_size=32, pattern_threshold=0.5, stability=5.0, seed=seed)
        self.decision_field = ThinkingField(field_size=32, pattern_threshold=0.55, stability=5.0, seed=seed)

        # State
        self.energy = config.initial_energy
        self.age = 0
        self.mating_cooldown = 0
        self.is_mating = False

        # Movement state
        self.current_velocity = 0.0
        self.current_rotation = 0.0

        # Enhanced movement
        self.hover_offset = random.uniform(0, 2 * np.pi)
        self.hover_time = 0
        self.random_direction = random.uniform(-1, 1)
        self.last_direction_change = 0

        # Nearby bugs (will be updated by GUI)
        self.nearby_bugs = []

        # Genetic traits
        if traits:
            self.traits = traits
        else:
            self.traits = {
                'vision_cone_angle': config.vision_cone_angle,
                'vision_cone_length': config.vision_cone_length,
                'max_speed': config.max_speed,
                'turn_speed': config.turn_speed,
                'size': random.uniform(3.0, 6.0),
                'shade': random.uniform(0.2, 0.8)
            }

        # Memory for motion detection
        self.memory = deque(maxlen=10)

    def inherit_traits(self, parent_traits: Dict, mutation_rate: float):
        """Inherit traits from parent with possible mutations"""
        child_traits = parent_traits.copy()
        for trait in child_traits:
            if random.random() < mutation_rate:
                # Apply a small mutation
                if trait == 'size':
                    mutation = np.random.normal(0, 0.2)
                    child_traits[trait] += mutation
                    child_traits[trait] = np.clip(child_traits[trait], 2.0, 10.0)
                elif trait == 'shade':
                    mutation = np.random.normal(0, 0.05)
                    child_traits[trait] += mutation
                    child_traits[trait] = np.clip(child_traits[trait], 0.0, 1.0)
                else:
                    mutation = np.random.normal(0, 0.05 * child_traits[trait])
                    child_traits[trait] += mutation
                    # Ensure traits stay within reasonable bounds
                    if trait == 'vision_cone_angle':
                        child_traits[trait] = np.clip(child_traits[trait], np.pi / 6, np.pi / 2)
                    elif trait == 'vision_cone_length':
                        child_traits[trait] = int(np.clip(child_traits[trait], 100, 200))
                    elif trait == 'max_speed':
                        child_traits[trait] = float(np.clip(child_traits[trait], 5.0, 25.0))
                    elif trait == 'turn_speed':
                        child_traits[trait] = float(np.clip(child_traits[trait], 0.1, 0.5))
        self.traits = child_traits

    def get_vision_cone(self, frame: np.ndarray) -> np.ndarray:
        """Extract the portion of the frame visible in the bug's vision cone and add bug markings"""
        height, width = frame.shape[:2]

        # Create mask for vision cone
        mask = np.zeros((height, width), dtype=np.uint8)

        # Calculate vision cone points
        angle = self.traits['vision_cone_angle']
        length = self.traits['vision_cone_length']

        # Calculate cone edges
        left_angle = self.angle - angle / 2
        right_angle = self.angle + angle / 2

        # Create polygon points for cone
        points = np.array([
            [int(self.x), int(self.y)],
            [int(self.x + length * np.cos(left_angle)), 
             int(self.y + length * np.sin(left_angle))],
            [int(self.x + length * np.cos(right_angle)), 
             int(self.y + length * np.sin(right_angle))]
        ], dtype=np.int32)

        # Create a vision frame that includes other bugs
        vision_frame = frame.copy()

        # Draw other bugs into the vision frame
        for other in self.nearby_bugs:
            if other != self:
                # Draw a bright spot for each bug
                color = (255, 0, 255) if other.is_mating else (255, 255, 255)
                cv2.circle(vision_frame, 
                          (int(other.x), int(other.y)), 
                          self.config.bug_visibility_size,
                          color, 
                          -1)  # filled circle

        # Draw filled cone on mask
        cv2.fillPoly(mask, [points], 255)

        # Apply mask to frame that includes bug markings
        cone_image = cv2.bitwise_and(vision_frame, vision_frame, mask=mask)

        # Convert to grayscale for brightness and motion detection
        cone_gray = cv2.cvtColor(cone_image, cv2.COLOR_BGR2GRAY)

        # Resize to desired resolution
        cone_image_resized = cv2.resize(cone_gray, 
                              (self.config.vision_width, self.config.vision_height))

        # Keep the image in grayscale (32,32)
        cone_image_gray = cone_image_resized
        return cone_image_gray

    def process_frame(self, frame: np.ndarray) -> Tuple[float, float]:
        """Process frame and decide movement using Thinking Fields"""
        try:
            # Extract vision cone
            cone_image = self.get_vision_cone(frame)

            # Integrate vision input into Perception Field
            self.perception_field.apply_input(cone_image)

            # Evolve the Perception Field
            self.perception_field.evolve()

            # Detect stable patterns in Perception Field
            perception_patterns = self.perception_field.detect_patterns()

            # Transfer relevant patterns to Memory Field
            self.memory_field.associate_patterns(perception_patterns)

            # Evolve the Memory Field
            self.memory_field.evolve()

            # Detect stable patterns in Memory Field
            memory_patterns = self.memory_field.detect_patterns()

            # Transfer patterns to Decision Field
            self.decision_field.associate_patterns(memory_patterns)

            # Evolve the Decision Field
            self.decision_field.evolve()

            # Detect stable patterns in Decision Field
            decision_patterns = self.decision_field.detect_patterns()

            # Generate decision based on Decision Patterns
            velocity_x, velocity_y, eating, mating = self.decision_field.generate_decision()

            # Convert velocity vector to speed and rotation
            velocity = np.sqrt(velocity_x**2 + velocity_y**2)
            rotation = np.arctan2(velocity_y, velocity_x) - self.angle

            # Update state
            self.is_mating = mating

            # Update energy based on brightness and motion
            self._update_energy(cone_image, want_eat=eating)

            # Apply momentum
            self.current_velocity = (self.current_velocity * self.config.momentum + 
                                   velocity * self.traits['max_speed'] * (1 - self.config.momentum))
            self.current_rotation = (self.current_rotation * self.config.momentum + 
                                   rotation * self.config.turn_speed * (1 - self.config.momentum))

            return self.current_velocity, self.current_rotation

        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")
            return 0.0, 0.0

    def _update_energy(self, vision_cone: np.ndarray, want_eat: bool):
        """Update bug's energy based on motion and light in camera feed"""
        # Base energy decay
        self.energy -= self.config.energy_decay

        if want_eat:
            # Calculate brightness in vision cone
            brightness = np.mean(vision_cone) / 255.0

            # Detect motion if we have previous frames
            motion_energy = 0
            if len(self.memory) > 0:
                prev_frame = self.memory[-1]
                diff = cv2.absdiff(vision_cone, prev_frame)
                motion = np.mean(diff) / 255.0
                motion_energy = motion * 10.0  # Scale up motion contribution

            # Energy gain based on environment
            energy_gain = (brightness * 2.0) + motion_energy

            # Cap maximum energy gain
            energy_gain = min(energy_gain, 5.0)

            self.energy += energy_gain

        # Store current frame for motion detection
        self.memory.append(vision_cone.copy())

        # Cap maximum energy
        self.energy = min(self.energy, self.config.initial_energy * 1.5)

    def calculate_repulsion(self) -> Tuple[float, float]:
        """Calculate repulsion vector from nearby bugs"""
        repulsion_x = 0
        repulsion_y = 0

        for other in self.nearby_bugs:
            if other != self:
                dx = self.x - other.x
                dy = self.y - other.y
                distance = np.sqrt(dx*dx + dy*dy)

                if distance < self.config.repulsion_distance and distance > 0:
                    # Calculate repulsion strength (stronger at closer distances)
                    strength = (1 - distance / self.config.repulsion_distance) * self.config.repulsion_force
                    # Add repulsion vector
                    repulsion_x += dx / distance * strength
                    repulsion_y += dy / distance * strength

        return repulsion_x, repulsion_y

    def try_mate(self, other: 'EnhancedBug') -> Optional['EnhancedBug']:
        """Attempt to mate with another bug, ensuring safe birth location and inheritance of thinking fields"""
        if (self.is_mating and other.is_mating and 
            self.energy > self.config.mating_threshold and 
            other.energy > self.config.mating_threshold and
            self.mating_cooldown == 0 and other.mating_cooldown == 0):
            
            # Introduce a 1/100 chance to mate
            if random.randint(1, 100) != 1:
                return None

            # Try to find a safe birth location (no obstacles to consider)
            safe_location = False
            max_attempts = 10
            attempt = 0

            while not safe_location and attempt < max_attempts:
                # Calculate potential birth location
                if attempt == 0:
                    # First try midpoint
                    child_x = (self.x + other.x) / 2
                    child_y = (self.y + other.y) / 2
                else:
                    # Then try random positions nearby
                    angle = random.uniform(0, 2 * np.pi)
                    distance = random.uniform(10, 30)
                    child_x = (self.x + other.x) / 2 + distance * np.cos(angle)
                    child_y = (self.y + other.y) / 2 + distance * np.sin(angle)

                safe_location = True
                attempt += 1

            # Create child at safe location
            child = EnhancedBug(
                x=child_x,
                y=child_y,
                config=self.config,
                traits={},  # Will be inherited
                seed=random.randint(0, 1_000_000)  # Fixed for Python 3.12
            )
            # Inherit traits from both parents with mutations
            child.inherit_traits(self.traits, self.config.mutation_rate)
            child.inherit_traits(other.traits, self.config.mutation_rate)

            # Blend Thinking Fields from parents
            child.perception_field.field = self.perception_field.blend_fields(other.perception_field)
            child.memory_field.field = self.memory_field.blend_fields(other.memory_field)
            child.decision_field.field = self.decision_field.blend_fields(other.decision_field)

            # Update parent states
            self.energy -= self.config.reproduction_cost
            other.energy -= self.config.reproduction_cost
            self.mating_cooldown = 100
            other.mating_cooldown = 100

            return child

        return None

    def update_position(self, velocity: float, rotation: float):
        """Update position with enhanced movement and repulsion"""
        # Get repulsion from nearby bugs
        repulsion_x, repulsion_y = self.calculate_repulsion()

        # Update hover
        self.hover_time += 0.1
        hover_y = np.sin(self.hover_time + self.hover_offset) * self.config.hover_amplitude

        # Random direction changes
        if self.age - self.last_direction_change > 20:
            if random.random() < self.config.random_movement:
                self.random_direction = random.uniform(-1, 1)
                self.last_direction_change = self.age

        # Add random movement to rotation
        rotation += self.random_direction * self.config.turn_speed

        # Limit rotation to prevent accumulation
        max_rotation = np.pi / 8  # 22.5 degrees
        rotation = np.clip(rotation, -max_rotation, max_rotation)

        # Update angle with enhanced rotation
        self.angle += rotation

        # Introduce damping to rotation to prevent accumulation
        self.current_rotation *= self.config.momentum

        # Calculate new position with hover effect and repulsion
        base_x = velocity * np.cos(self.angle)
        base_y = velocity * np.sin(self.angle) + hover_y

        # Add repulsion to movement
        self.x += base_x + repulsion_x
        self.y += base_y + repulsion_y

        # Keep bug on screen by wrapping around edges
        if hasattr(self, 'screen_width') and hasattr(self, 'screen_height'):
            self.x = self.x % self.screen_width
            self.y = self.y % self.screen_height

        # Update state
        self.age += 1
        if self.mating_cooldown > 0:
            self.mating_cooldown -= 1

# -----------------------------
# BugGUI Class
# -----------------------------

class BugGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Evolving Flies Simulation")

        # Create config and setup controls first
        self.config = BugConfig()
        self.setup_camera_selection()

        # Initialize camera
        self.initialize_camera()

        # Create bugs
        self.bugs: List[EnhancedBug] = []
        self.fly_visuals = FlyVisuals()
        self.initialize_population()

        # Setup rest of GUI elements
        self.setup_gui()

        # Processing queues
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = True
        self.paused = False

        # Statistics
        self.generation = 1
        self.total_births = 0
        self.total_deaths = 0
        self.population_history = [self.config.initial_population]
        self.start_time = time.time()

        # Visual effects
        self.visual_effects: List[VisualEffect] = []

        # Data export
        self.setup_data_export()

        # Start processing
        self.start_processing()

    def setup_data_export(self):
        """Setup CSV file for exporting statistics"""
        self.csv_file = "simulation_stats.csv"
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time", "Population", "Births", "Deaths"])

    def export_data(self):
        """Export current statistics to CSV"""
        elapsed = time.time() - self.start_time
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"{elapsed:.2f}", len(self.bugs), self.total_births, self.total_deaths])

    def initialize_population(self):
        """Create initial bug population with better distribution"""
        self.bugs: List[EnhancedBug] = []
        margin = 50  # Keep away from edges
        grid_size = int(np.ceil(np.sqrt(self.config.initial_population)))
        spacing_x = (self.width - 2 * margin) / grid_size
        spacing_y = (self.height - 2 * margin) / grid_size

        for i in range(self.config.initial_population):
            # Calculate grid position
            grid_x = i % grid_size
            grid_y = i // grid_size

            # Add some random offset
            x = margin + grid_x * spacing_x + random.uniform(-spacing_x/4, spacing_x/4)
            y = margin + grid_y * spacing_y + random.uniform(-spacing_y/4, spacing_y/4)

            bug = EnhancedBug(
                x=x, 
                y=y, 
                config=self.config,
                seed=random.randint(0, 1_000_000)  # Fixed for Python 3.12
            )
            bug.screen_width = self.width
            bug.screen_height = self.height
            self.bugs.append(bug)

    def setup_camera_selection(self):
        """Create camera selection dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Camera Selection")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Select Camera:").pack(padx=10, pady=5)

        # Create camera list
        camera_frame = ttk.Frame(dialog)
        camera_frame.pack(fill=tk.X, padx=10, pady=5)

        # Try first 5 camera indices
        self.available_cameras = []
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.available_cameras.append(i)
                    cap.release()
            except:
                continue

        if not self.available_cameras:
            messagebox.showerror("Error", "No cameras found!")
            sys.exit(1)

        # Camera selection dropdown
        self.camera_var = tk.IntVar(value=self.available_cameras[0])
        camera_select = ttk.Combobox(
            camera_frame, 
            textvariable=self.camera_var,
            values=self.available_cameras,
            state='readonly',
            width=10
        )
        camera_select.pack(side=tk.LEFT, padx=5)

        # Resolution selection
        resolution_frame = ttk.Frame(camera_frame)
        resolution_frame.pack(side=tk.LEFT, padx=10)

        resolutions = ['640x480', '800x600', '1280x720', '1920x1080']
        self.resolution_var = tk.StringVar(value='1280x720')
        ttk.Label(resolution_frame, text="Resolution:").pack(side=tk.LEFT)
        resolution_select = ttk.Combobox(
            resolution_frame,
            textvariable=self.resolution_var,
            values=resolutions,
            state='readonly',
            width=12
        )
        resolution_select.pack(side=tk.LEFT, padx=5)

        # Test button
        def test_camera():
            try:
                cap = cv2.VideoCapture(self.camera_var.get())
                if cap.isOpened():
                    # Set resolution
                    w, h = map(int, self.resolution_var.get().split('x'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

                    ret, frame = cap.read()
                    if ret:
                        # Show small preview
                        frame = cv2.resize(frame, (320, 240))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)
                        photo = ImageTk.PhotoImage(image)

                        preview = tk.Toplevel(dialog)
                        preview.title("Camera Preview")
                        label = ttk.Label(preview, image=photo)
                        label.image = photo
                        label.pack()

                        def close_preview():
                            cap.release()
                            preview.destroy()

                        preview.after(2000, close_preview)
                    cap.release()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to test camera: {str(e)}")

        ttk.Button(camera_frame, text="Test", command=test_camera).pack(side=tk.LEFT, padx=5)

        # OK button
        def on_ok():
            self.config.camera_index = self.camera_var.get()
            w, h = map(int, self.resolution_var.get().split('x'))
            self.config.camera_width = w
            self.config.camera_height = h
            dialog.destroy()

        ttk.Button(dialog, text="OK", command=on_ok).pack(pady=10)

        # Wait for dialog
        self.root.wait_window(dialog)

    def initialize_camera(self):
        """Initialize selected camera"""
        try:
            self.camera = cv2.VideoCapture(self.config.camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(f"Could not open camera {self.config.camera_index}")

            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

            # Get actual resolution (explicit integer conversion)
            self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"Camera initialized at resolution: {self.width}x{self.height}")

        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to initialize camera: {str(e)}")
            sys.exit(1)

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel at top
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Left side controls
        left_controls = ttk.Frame(control_frame)
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Stats frame
        stats_frame = ttk.LabelFrame(left_controls, text="Statistics", padding="5")
        stats_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.population_var = tk.StringVar(value="Population: 0")
        self.generation_var = tk.StringVar(value="Generation: 1")
        self.births_var = tk.StringVar(value="Births: 0")
        self.deaths_var = tk.StringVar(value="Deaths: 0")

        ttk.Label(stats_frame, textvariable=self.population_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(stats_frame, textvariable=self.generation_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(stats_frame, textvariable=self.births_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(stats_frame, textvariable=self.deaths_var).pack(side=tk.LEFT, padx=5)

        # Camera selection button
        def change_camera():
            self.cleanup()
            self.setup_camera_selection()
            self.initialize_camera()
            self.start_processing()

        ttk.Button(stats_frame, text="Change Camera", command=change_camera).pack(side=tk.LEFT, padx=5)

        # Right side controls
        right_controls = ttk.Frame(control_frame)
        right_controls.pack(side=tk.RIGHT, fill=tk.X)

        # Parameters frame
        params_frame = ttk.LabelFrame(right_controls, text="Parameters", padding="5")
        params_frame.pack(side=tk.RIGHT, fill=tk.X, padx=5)

        # Energy decay slider
        ttk.Label(params_frame, text="Energy Decay:").pack(side=tk.LEFT)
        self.energy_scale = ttk.Scale(
            params_frame, 
            from_=0.01, 
            to=0.2,
            orient=tk.HORIZONTAL,
            length=100,
            value=self.config.energy_decay,
            command=self.update_energy_decay
        )
        self.energy_scale.pack(side=tk.LEFT, padx=5)

        # Speed slider
        ttk.Label(params_frame, text="Max Speed:").pack(side=tk.LEFT)
        self.speed_scale = ttk.Scale(
            params_frame, 
            from_=5.0, 
            to=25.0,
            orient=tk.HORIZONTAL,
            length=100,
            value=self.config.max_speed,
            command=self.update_max_speed
        )
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        # Population slider
        ttk.Label(params_frame, text="Max Pop:").pack(side=tk.LEFT)
        self.pop_scale = ttk.Scale(
            params_frame, 
            from_=10, 
            to=200,
            orient=tk.HORIZONTAL,
            length=100,
            value=self.config.max_population,
            command=self.update_max_population
        )
        self.pop_scale.pack(side=tk.LEFT, padx=5)

        # Save and Load buttons
        ttk.Button(right_controls, text="Save Population", command=self.save_population).pack(side=tk.RIGHT, padx=5)
        ttk.Button(right_controls, text="Load Population", command=self.load_population).pack(side=tk.RIGHT, padx=5)

        # Pause/Resume button
        self.pause_button = ttk.Button(right_controls, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side=tk.RIGHT, padx=5)

        # Population trend graph
        graph_frame = ttk.LabelFrame(main_frame, text="Population Trend", padding="5")
        graph_frame.pack(fill=tk.X, padx=5, pady=5)

        self.figure = Figure(figsize=(5, 2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Population Over Time")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Population")
        self.line, = self.ax.plot([], [], 'b-')
        self.population_times = []
        self.population_values = []
        self.canvas_graph = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas_graph.draw()
        self.canvas_graph.get_tk_widget().pack()

        # Canvas for video display
        self.canvas = tk.Canvas(main_frame, width=self.width, height=self.height, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def update_energy_decay(self, val):
        self.config.energy_decay = float(val)

    def update_max_speed(self, val):
        self.config.max_speed = float(val)

    def update_max_population(self, val):
        self.config.max_population = int(float(val))

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")

    def start_processing(self):
        """Start processing threads"""
        self.running = True

        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

        # Start processing loop
        self.process_frame()

    def camera_loop(self):
        """Background thread for camera capture"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            time.sleep(1/30)  # Limit to 30fps

    def process_frame(self):
        """Main processing loop"""
        if self.paused:
            self.root.after(100, self.process_frame)
            return

        try:
            if not self.frame_queue.empty():
                # Get frame from queue
                frame = self.frame_queue.get()

                # Update parameters from sliders
                self.config.energy_decay = self.energy_scale.get()
                self.config.max_speed = self.speed_scale.get()
                self.config.max_population = int(self.pop_scale.get())

                # Convert frame to grayscale for motion detection (not used directly)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Process all bugs
                dead_bugs = []
                new_bugs = []

                # Spatial partitioning: simple grid-based
                grid_size = 100
                grid = {}
                for bug in self.bugs:
                    grid_x = int(bug.x // grid_size)
                    grid_y = int(bug.y // grid_size)
                    grid.setdefault((grid_x, grid_y), []).append(bug)

                for bug in self.bugs:
                    # Find nearby bugs using grid
                    grid_x = int(bug.x // grid_size)
                    grid_y = int(bug.y // grid_size)
                    nearby = []
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            cell = (grid_x + dx, grid_y + dy)
                            if cell in grid:
                                nearby.extend(grid[cell])
                    bug.nearby_bugs = [other for other in nearby if other != bug and 
                                       np.sqrt((bug.x - other.x)**2 + (bug.y - other.y)**2) < self.config.bug_detection_range]

                    # Process through bug's thinking fields
                    velocity, rotation = bug.process_frame(frame)

                    # Update bug position
                    bug.update_position(velocity, rotation)

                    # Check for death
                    if bug.energy <= 0:
                        dead_bugs.append(bug)
                        self.total_deaths += 1
                        # Add death effect
                        self.visual_effects.append(
                            VisualEffect(bug.x, bug.y,
                                       self.config.death_marker_duration,
                                       'death')
                        )

                    # Check for reproduction
                    if (bug.is_mating and len(self.bugs) < self.config.max_population):
                        for other in bug.nearby_bugs:
                            if other.is_mating and other not in new_bugs:
                                child = bug.try_mate(other)
                                if child:
                                    child.screen_width = self.width
                                    child.screen_height = self.height
                                    new_bugs.append(child)
                                    self.total_births += 1
                                    # Add birth effect
                                    self.visual_effects.append(
                                        VisualEffect(child.x, child.y,
                                                   self.config.birth_marker_duration,
                                                   'birth')
                                    )
                                    break

                # Remove dead bugs and add new ones
                for bug in dead_bugs:
                    self.bugs.remove(bug)
                self.bugs.extend(new_bugs)

                # Repopulate if count dropped below minimum
                if len(self.bugs) < self.config.min_population:
                    self.repopulate()

                # Update population history
                current_time = time.time() - self.start_time
                self.population_times.append(current_time)
                self.population_values.append(len(self.bugs))
                self.population_history.append(len(self.bugs))

                # Update display
                self.update_display(frame)

                # Update statistics
                self.update_stats()

                # Export data periodically
                if int(current_time) % 10 == 0:
                    self.export_data()

            # Update population graph
            self.update_graph()

            # Schedule next update
            self.root.after(33, self.process_frame)

        except Exception as e:
            logging.error(f"Processing error: {e}")
            messagebox.showerror("Error", str(e))

    def update_stats(self):
        """Update statistics display"""
        self.population_var.set(f"Population: {len(self.bugs)}")
        self.generation_var.set(f"Generation: {self.generation}")
        self.births_var.set(f"Births: {self.total_births}")
        self.deaths_var.set(f"Deaths: {self.total_deaths}")

    def repopulate(self):
        """Add new bugs when population is too low"""
        while len(self.bugs) < self.config.initial_population:
            bug = EnhancedBug(
                x=random.randint(0, int(self.width)),
                y=random.randint(0, int(self.height)),
                config=self.config,
                seed=random.randint(0, 1_000_000)  # Fixed for Python 3.12
            )
            bug.screen_width = self.width
            bug.screen_height = self.height
            self.bugs.append(bug)

    def update_display(self, frame):
        """Update display with flies and effects"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw all bugs
        for bug in self.bugs:
            self.fly_visuals.draw_fly(
                frame_rgb,
                bug.x,
                bug.y,
                bug.angle,
                bug.is_mating,
                bug.traits['size'],
                bug.traits['shade']
            )

        # Update and draw visual effects
        remaining_effects = []
        for effect in self.visual_effects:
            if effect.effect_type == 'birth':
                effect.draw(frame_rgb)
            elif effect.effect_type == 'death':
                effect.draw(frame_rgb)
            if effect.update():
                remaining_effects.append(effect)
        self.visual_effects = remaining_effects

        # Convert to PhotoImage
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)

        # Update canvas
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.photo = photo

    def update_graph(self):
        """Update the population trend graph"""
        self.ax.clear()
        self.ax.set_title("Population Over Time")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Population")
        self.ax.plot(self.population_times, self.population_values, 'b-')
        self.canvas_graph.draw()

    def save_population(self):
        """Save the current population to a JSON file"""
        if not self.bugs:
            messagebox.showwarning("Save Population", "No bugs to save.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save Population"
        )
        if not file_path:
            return
        try:
            data = []
            for bug in self.bugs:
                bug_data = {
                    'x': bug.x,
                    'y': bug.y,
                    'angle': bug.angle,
                    'energy': bug.energy,
                    'age': bug.age,
                    'mating_cooldown': bug.mating_cooldown,
                    'is_mating': bug.is_mating,
                    'traits': bug.traits,
                    'perception_field': bug.perception_field.copy_field().tolist(),
                    'memory_field': bug.memory_field.copy_field().tolist(),
                    'decision_field': bug.decision_field.copy_field().tolist(),
                    'seed': getattr(bug, 'perception_field', None)
                }
                data.append(bug_data)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            messagebox.showinfo("Save Population", f"Population saved to {file_path}")
        except Exception as e:
            logging.error(f"Save population error: {e}")
            messagebox.showerror("Save Population Error", str(e))

    def load_population(self):
        """Load a population from a JSON file"""
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Load Population"
        )
        if not file_path:
            return
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            loaded_bugs = []
            for bug_data in data:
                bug = EnhancedBug(
                    x=bug_data['x'],
                    y=bug_data['y'],
                    config=self.config,
                    traits=bug_data['traits'],
                    seed=bug_data.get('seed', random.randint(0, 1_000_000))
                )
                bug.angle = bug_data['angle']
                bug.energy = bug_data['energy']
                bug.age = bug_data['age']
                bug.mating_cooldown = bug_data['mating_cooldown']
                bug.is_mating = bug_data['is_mating']
                bug.screen_width = self.width
                bug.screen_height = self.height

                # Restore Thinking Fields
                bug.perception_field.field = np.array(bug_data['perception_field'])
                bug.memory_field.field = np.array(bug_data['memory_field'])
                bug.decision_field.field = np.array(bug_data['decision_field'])

                loaded_bugs.append(bug)
            self.bugs = loaded_bugs
            self.population_history = [len(self.bugs)]
            self.population_times = [time.time() - self.start_time]
            self.population_values = [len(self.bugs)]
            messagebox.showinfo("Load Population", f"Population loaded from {file_path}")
        except Exception as e:
            logging.error(f"Load population error: {e}")
            messagebox.showerror("Load Population Error", str(e))

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if hasattr(self, 'camera_thread') and self.camera_thread is not None:
            self.camera_thread.join()
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()

# -----------------------------
# Main Function
# -----------------------------

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("flies.log"),
            logging.StreamHandler()
        ]
    )

    # Create main window
    root = tk.Tk()

    try:
        # Create and run application
        app = BugGUI(root)

        # Setup cleanup
        def on_closing():
            app.cleanup()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        # Start main loop
        root.mainloop()

    except Exception as e:
        logging.error(f"Application error: {e}")
        messagebox.showerror("Error", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
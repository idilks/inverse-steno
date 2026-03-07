# Inverse Planning Agent Data Generator for Steganography Detection

## Overview

This repository implements the **attacker-side data generation pipeline** for an inverse-planning steganography detection project.

The purpose of this code is **not** to implement the inverse-planning detector itself. Instead, it generates datasets containing:

- **Honest responses** (task-only optimization)
- **Steganographic responses** (task + hidden channel optimization)

These paired outputs allow a downstream inverse-planning detector to analyze the **steganography tax** — the statistical distortion introduced when an agent must satisfy both a task objective and a hidden communication objective.

---

# Core Idea

An honest agent optimizes only for solving the cover task.

A steganographic agent optimizes for:

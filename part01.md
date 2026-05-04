# Project 02 - Dataset Proposal (Part 01)

## Goal
For this proposal, I came up with three possible datasets I could create for Project 02.

## Option 1 - weather scene image classification dataset (YOLO)
I can build a small image classification dataset with classes like:
- `clear_sky`
- `cloudy`
- `rain`
- `snow`
- `storm`

### i could build it by
- collecting images from free/open sources (or my own photos).
- organizing images into YOLO classification folder format:
  - `dataset/train/<class_name>/...`
  - `dataset/test/<class_name>/...`
- applying basic augmentation (crop, brightness, horizontal flip).

## Option 2 - Iowa City daily weather time-series dataset (Tabular ML)
i can build a tabular dataset from Open-Meteo historical daily data for Iowa City. this would fit in with my thesis

### example columns
- `temperature_2m_max`
- `temperature_2m_min`
- `precipitation_sum`
- `windspeed_10m_max`
- `weather_code`
- engineered features like lag values and rolling means

### target ideas
- regression: next-day max temperature
- classification: storm day vs non-storm day

## Option 3 - weather + housing risk dataset
i can create a joined dataset that combines:
- historical weather features (Open-Meteo)
- simplified housing/material assumptions (for exposure/risk scoring)

### output idea
A yearly or seasonal "impact score" dataset for property risk categories (`low`, `medium`, `high`).

## Preferred Choice
My preferred direction is option 2, and option 3 if i have time

I already drafted a standalone script in this repository (`project02_weather_model.py`) that downloads weather data, engineers ML features, trains both regression and classification baselines, and saves a local dataset file.



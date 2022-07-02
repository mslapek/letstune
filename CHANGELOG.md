# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- `letstune.keras.KerasTrainer` calls `model.fit` with turned off Keras logging (`verbose=0`). 

## [0.2.0] - 2022-06-24
### Added
- Trainers can return `np.float32` and `np.float64` in `metric_values`. Backend gives a friendly error message in case of malformed `metric_values`.

### Changed
- Replace `training_maximum_duration` in `letstune.tune` with `rounds` parameter.

## [0.1.0] - 2022-05-22
### Added
- First version :tada: with local tuning by [@mslapek](https://github.com/mslapek).

[Unreleased]: https://github.com/mslapek/letstune/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mslapek/letstune/releases/tag/v0.2.0
[0.1.0]: https://github.com/mslapek/letstune/releases/tag/v0.1.0

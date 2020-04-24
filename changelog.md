# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed
- `chess pairs`: Fix error on empty lines in chromosome sizes file.
- `chess pairs`: Catch OSError raised by pybedtools when the provided path is not recognized as a UCSC genome id.

## [0.2.0]

### Added
- `chess extract`: Extract specific regions that are significantly different
- `chess crosscorrelate`: Get structural clusters from the extracted submatrices
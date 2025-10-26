# Import settings
from app.config import settings

# Use in application
if settings.is_production:
    # Production-specific code
    pass

# Check configuration
warnings = settings.validate_configuration()
if warnings:
    for warning in warnings:
        logger.warning(warning)

# Get safe configuration for logging
safe_config = settings.get_safe_dict()
logger.info(f"Configuration: {safe_config}")

# Generate .env template
settings.write_env_template(".env.example")

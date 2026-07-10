# Use the highly optimized, lightweight community build of BirdNET
FROM ghcr.io/tidepool-labs/birdnet-go:latest

# Expose the web server port
EXPOSE 8080

# Run the analyzer server on startup
CMD ["server", "--port", "8080"]

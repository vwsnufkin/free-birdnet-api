# Use the official, updated community build of BirdNET
FROM ghcr.io/tphakala/birdnet-go:latest

# Expose the web server port
EXPOSE 8080

# Run the analyzer server on startup
CMD ["birdnet-go", "server", "--port", "8080"]

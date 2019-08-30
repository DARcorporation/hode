docker build --no-cache -t hode-example .
docker run -e np=4 dim=2 bits=31 hode-example
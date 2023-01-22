.PHONY: all start attach stop restart
CONFIG=config/config.env
include ${CONFIG}                                                                                    

all: start 
start:
	docker-compose --env-file ${CONFIG} up webserver nginx infer infer_cpp_httplib infer_cpp_crow
build:
	docker-compose --env-file ${CONFIG} build
attach:
	docker attach ${CONTAINER}
stop: 
	docker-compose --env-file ${CONFIG} kill
# remove:
# 	docker-
restart: stop start
re: build start

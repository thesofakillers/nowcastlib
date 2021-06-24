document:
	@rm -rf docs/*
	@pdoc --html  --force -c sort_identifiers=False -o docs nowcastlib 
	@mv docs/nowcastlib/* docs
	@rmdir docs/nowcastlib

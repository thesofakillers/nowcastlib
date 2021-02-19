document:
	@pdoc --html  --force -o docs nowcastlib 
	@mv docs/nowcastlib/* docs
	@rmdir docs/nowcastlib

all: html

build/%.ipynb: %.md build/build.yml
	cd build; python md2ipynb.py ../$< ../$@

build/%: %
	@mkdir -p $(@D)
	@cp -r $< $@

MARKDOWN = index.rst mxnet_packages.md use_aws.md preface.md
NOTEBOOK = ndarray.md nn.md autograd.md train.md predict.md use_gpus.md

OBJ = $(patsubst %, build/%, $(MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, $(NOTEBOOK))

ORIGN_DEPS = $(wildcard img/*) README.md
DEPS = $(patsubst %, build/%, $(ORIGN_DEPS))

PKG = build/_build/html/gluon_crash_course.tar.gz build/_build/html/gluon_crash_course.zip

pkg: $(PKG)

build/_build/html/gluon_crash_course.zip: $(OBJ) $(DEPS)
	cd build; zip -r $(patsubst build/%, %, $@ $(DEPS)) *.ipynb

build/_build/html/gluon_crash_course.tar.gz: $(OBJ) $(DEPS)
	cd build; tar -zcvf $(patsubst build/%, %, $@ $(DEPS)) *.ipynb

html: $(DEPS) $(OBJ)
	make -C build html

TEX=build/_build/latex/gluon_crash_course.tex

pdf: $(DEPS) $(OBJ)
	make -C build latex
	# sed -i s/{tocdepth}{0}/{tocdepth}{1}/ $(TEX)
	cd build/_build/latex && \
	buf_size=10000000 xelatex gluon_crash_course.tex && \
	buf_size=10000000 xelatex gluon_crash_course.tex

clean:
	rm -rf $(DEPS) $(PKG) build/*.ipynb

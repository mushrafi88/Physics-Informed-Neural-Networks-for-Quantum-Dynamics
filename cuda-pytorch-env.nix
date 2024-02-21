{ pkgs ? import <nixpkgs> { } }:
let
  fhs = pkgs.buildFHSUserEnv {
    name = "cuda-pytorch environment";

    targetPkgs = pkgs: with pkgs; [
      git
      gitRepo
      gnupg
      autoconf
      curl
      procps
      gnumake
      util-linux
      m4
      gperf
      unzip
      cudatoolkit
      linuxPackages.nvidia_x11
      libGLU
      libGL
      xorg.libXi
      xorg.libXmu
      freeglut
      xorg.libXext
      xorg.libX11
      xorg.libXv
      xorg.libXrandr
      zlib
      ncurses5
      stdenv.cc
      binutils
      micromamba
    ];
    multiPkgs = pkgs: with pkgs; [ zlib ];
    runScript = "bash";
    profile = ''
        set -e
        export CUDA_PATH=${pkgs.cudatoolkit}
        # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
        export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        export EXTRA_CCFLAGS="-I/usr/include"
        export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba
        eval "$(micromamba shell hook --shell=bash | sed 's/complete / # complete/g')"
        if micromamba list -n pytorch > /dev/null 2>&1; then
            echo "Activating existing 'pytorch' environment."
            micromamba activate pytorch
            jupyter lab
        else
            echo "Creating new 'pytorch' environment."
            micromamba create --yes -q -n pytorch
            micromamba activate pytorch
            micromamba install pytorch torchvision torchaudio pytorch-cuda=11.8 jupyter -c pytorch -c nvidia -c conda-forge
            micromamba install -c conda-forge ipympl
            pip install matplotlib pandas seaborn scikit-learn tqdm 
            jupyter lab
        fi
        set +e
    '';
  };
in
fhs.env

{
  description = "An FHS shell with mamba and cuda and miniconda for safety test.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, home-manager }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in {
      devShell.x86_64-linux = (pkgs.buildFHSUserEnv {
        name = "conda";
        targetPkgs = pkgs: (
          with pkgs; [
            autoconf
            binutils
            cudatoolkit
            curl
            freeglut
            gcc11
            git
            gitRepo
            gnumake
            gnupg
            gperf
            libGLU 
            libGL
            libselinux
            linuxPackages.nvidia_x11
            m4
            ncurses5
            procps
            stdenv.cc
            unzip
            util-linux
            wget
            xorg.libICE
            xorg.libSM
            xorg.libX11
            xorg.libXext
            xorg.libXi
            xorg.libXmu
            xorg.libXrandr
            xorg.libXrender
            xorg.libXv
            zlib
            micromamba
          ]
        );
        profile = ''
          # cuda
          export CUDA_PATH=${pkgs.cudatoolkit}
          # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
          export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
          export EXTRA_CCFLAGS="-I/usr/include" 
          export FONTCONFIG_FILE=/etc/fonts/fonts.conf
          export QTCOMPOSE=${pkgs.xorg.libX11}/share/X11/locale

          #micromamba 
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
            micromamba install -c conda-forge jupyter-ai
            pip install matplotlib pandas seaborn scikit-learn tqdm catppuccin-jupyterlab 
            jupyter lab
          fi
        '';
      }).env;
    };
}

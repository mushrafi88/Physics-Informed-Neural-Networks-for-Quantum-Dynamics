{
  description = "A Nix Flake for QuantumPINN with CUDA-PyTorch environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true; # Allow unfree packages
          };
          overlays = [
            (self: super: {
              cuda-pytorch-env = super.callPackage ./cuda-pytorch-env.nix { };
            })
          ];
        };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [ pkgs.cuda-pytorch-env ];
        };
      });
}


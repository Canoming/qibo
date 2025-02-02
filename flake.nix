{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
  };

  outputs = {
    self,
    nixpkgs,
    devenv,
    systems,
    ...
  } @ inputs: let
    forEachSystem = nixpkgs.lib.genAttrs (import systems);
  in {
    packages = forEachSystem (system: {
      default =
        nixpkgs.legacyPackages.${system}.poetry2nix.mkPoetryApplication
        {
          projectDir = self;
          preferWheels = true;
        };
    });

    devShells =
      forEachSystem
      (system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        default = devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [
            {
              packages = with pkgs; [
                pre-commit
                (python3Packages.buildPythonPackage
                  (let
                    pname = "poethepoet";
                    version = "0.24.1";
                  in {
                    inherit pname version;
                    src = fetchPypi {
                      inherit pname version;
                      sha256 = "sha256-OvpEtPxzJ98N2RLtoBJgSgcq8rtNJD+w5B6Oyo2r+e0=";
                      python = "py3";
                      dist = "py3";
                      format = "wheel";
                    };
                    format = "wheel";
                    propagatedBuildInputs = [
                      # Specify dependencies
                      pkgs.python3Packages.pastel
                      pkgs.python3Packages.tomli
                    ];
                  }))
              ];

              languages.python = {
                enable = true;
                poetry = {
                  enable = true;
                  install.enable = true;
                  install.allExtras = true;
                  install.groups = ["dev"];
                };
                version = "3.11";
              };
            }
          ];
        };
      });
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };
}

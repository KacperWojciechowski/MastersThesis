{
  inputs = {
    nixpkgs = { url = "nixpkgs/nixpkgs-unstable"; };
    flake-utils-plus = { url = "github:gytis-ivaskevicius/flake-utils-plus"; };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils-plus,
    ...
  } @ inputs: let
    inherit (flake-utils-plus.lib) defaultSystems eachDefaultSystem;

    lib = nixpkgs.lib;
  in
    (eachDefaultSystem (system: 
      let
        inherit (pkgs) stdenv;

        pkgs = nixpkgs.legacyPackages.${system};
        mkShell = pkgs.mkShell;
        shogun = (pkgs.shogun.overrideAttrs (_: { doCheck = false; }));

        buildDeps = with pkgs; [ 
          pkgconfig
          cmake
          extra-cmake-modules
        ];

        appDeps = with pkgs; [
          shogun
          shogun.dev
	  dlib
	 ];

        testCpp = stdenv.mkDerivation {
          pname = "test-cpp";
          version = "0.0.1";

          src = ./Result;

          nativeBuildInputs = buildDeps;
          buildInputs = appDeps;
        };
      in {
        packages = {
          inherit testCpp;
        };

        devShells = {
          default = mkShell {
            buildInputs = with pkgs; [
              cmake
              extra-cmake-modules
            ]
              ++ buildDeps
              ++ appDeps;
          };
        };
      }));
}

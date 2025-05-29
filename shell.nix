{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {

  buildInputs = [
    pkgs.git
    (pkgs.python3.withPackages (ps: with ps; [
      numpy
      scipy
      matplotlib
      scikit-learn
      networkx
    ]))
  ];

}

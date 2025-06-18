#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvc.H"
#include "fvm.H"
#include "adjustPhi.H"
#include "argList.H"
#include "timeSelector.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::addNote("Corrects phi field from U and p using adjustPhi.");
    argList::noParallel();

    argList::addOption("time", "Select specific time directory");
    argList::addBoolOption("latestTime", "Select latest available time directory");

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    if (args.optionFound("help"))
    {
        Info<< "Usage: adjustPhiML [OPTIONS]\n\n"
            << "OPTIONS:\n"
            << "    -latestTime         Adjust phi at latest time directory.\n"
            << "    -time <time>         Adjust phi at a specific time directory.\n"
            << "    -help                Display this help message.\n"
            << endl;
        return 0;
    }

    instantList timeDirs = timeSelector::select0(runTime, args);

    forAll(timeDirs, timeI)
    {
        runTime.setTime(timeDirs[timeI], timeI);
        // Info << "Time = " << Time::timeName(runTime.value()) << nl << endl;

        volVectorField U(IOobject("U", Time::timeName(runTime.value()), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh);
        volScalarField p(IOobject("p", Time::timeName(runTime.value()), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh);
        volScalarField rho(IOobject("rho", Time::timeName(runTime.value()), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh);

        Info << "Running adjustPhi(rho * U)..." << endl;
        surfaceScalarField phi(IOobject("phi", Time::timeName(runTime.value()), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE), fvc::flux(rho * U));
        // surfaceScalarField phi(IOobject("phi",runTime.name(),mesh,IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE),linearInterpolate(rho * U) & mesh.Sf());

        adjustPhi(phi, U, p);

        phi.write();
        Info << "Corrected phi written to time " << Time::timeName(runTime.value()) << "\n" << endl;
    }

    Info << "End" << endl;
    return 0;
}

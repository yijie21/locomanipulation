using UnrealBuildTool;

public class URLabHostTarget : TargetRules
{
	public URLabHostTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("URLabHost");
	}
}

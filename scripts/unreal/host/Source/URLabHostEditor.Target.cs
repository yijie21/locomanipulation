using UnrealBuildTool;

public class URLabHostEditorTarget : TargetRules
{
	public URLabHostEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("URLabHost");
	}
}
